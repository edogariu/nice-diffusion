import enum
import math
import torch
import numpy as np
import tqdm

'''
DIFFUSION AND DENOISING UTILITIES BASED ON THE FOLLOWING PAPERS AND CORRESPONDING WORK:
    - Ho et al. Denoising Diffusion Probabilistic Models (DDPM): https://arxiv.org/pdf/2006.11239.pdf
    - Song et al. Denoising Diffusion Implicit Models (DDIM): https://arxiv.org/pdf/2010.02502.pdf
    - Dhariwal/Nichol Improved Denoising Diffusion Probabilistic Models (IDDPM): https://arxiv.org/pdf/2102.09672.pdf
    - Dhariwal/Nichol Diffusion Model Beats GAN on Image Synthesis (OpenAI): https://arxiv.org/pdf/2105.05233.pdf
    - Ho/Salismans Classifier-Free Diffusion Guidance (CFDG): 
                    https://openreview.net/pdf/ea628d03c92a49b54bc2d757d209e024e7885980.pdf

possible improvements: 
    - during sampling, denoise fully, then diffuse to step s, then denoise again starting from s
    - truncate if using a CDM setup
'''


class Diffusion:
    """
    Creates an object to handle a diffusion chain and a reverse diffusion (denoising) chain, with or without DDIM
    sampling.

        Parameters:
            - model (DiffusionModel): trained model to predict epsilon from noisy image
            - original_num_steps (int): number of diffusion steps that model was trained with (T)
            - rescaled_num_steps (int): number of diffusion steps to be considered when sampling
            - sampling_var_type (str): type of variance calculation -- 'small' or 'large' for fixed variances of given
              sizes, 'learned' (outputs log(var)) or 'learned_interpolation' (outputs interpolation value for log(var))
              for variances predicted by model
            - loss_type (str): type of training loss calculation -- 'simple' for simple MSE loss of prediction,
              'KL' or 'KL_rescaled' for loss using variational lower bound, 'hybrid' for weighted
              sum of simple and KL losses (note: KL is only used for learning vars in hybrid loss).
              If sampling_var_type is fixed (i.e. 'small' or 'large'), you should only use 'simple' loss
            - beta_schedule (str): scheduling method for noise variances (betas) -- 'linear', 'constant', or 'cosine'
            - betas (np.array): alternative to beta_schedule where betas are directly supplied
            - guidance_method (str): method of denoising guidance to use -- None, 'classifier', or 'classifier_free'
            - classifier (nn.Module): if guidance_method == 'classifier', which classifier model to use
            - guidance_strength (double): if guidance_method is not None, controls strength of guidance method selected
            - use_ddim (bool): whether to use DDIM sampling and interpret rescaled_num_steps as number of DDIM steps
            - ddim_eta (double): value to be used when performing DDIM
            - device (torch.device): if not None, which device to perform diffusion with


        Returns:
            - Diffusion object to call .diffuse(), .denoise(), or .loss() with with.
    """

    def __init__(self, model,
                 original_num_steps, rescaled_num_steps,
                 sampling_var_type, loss_type,
                 betas=None, beta_schedule='linear',
                 guidance_method=None, guidance_strength=None, classifier=None,
                 use_ddim=False, ddim_eta=None, device=None):
        self.model = model
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.guidance = guidance_method
        if guidance_method != 'classifier' and guidance_method != 'classifier_free' and guidance_method is not None:
            raise NotImplementedError(guidance_method)
        assert guidance_method is None or self.model.conditional, 'can only use guidance if model is conditional'
        self.strength = guidance_strength
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.float().to(self.device)
            self.classifier.eval()

        self.original_num_steps = original_num_steps
        self.rescaled_num_steps = rescaled_num_steps
        self.sampling_var_type = VarType.get_var_type(sampling_var_type)
        self.loss_type = LossType.get_loss_type(loss_type)

        if use_ddim:
            assert ddim_eta is not None, 'please supply eta if you want to use ddim'
        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta

        if betas is None:
            betas = get_beta_schedule(beta_schedule, original_num_steps,
                                      0.0001 * 1000 / original_num_steps, 0.02 * 1000 / original_num_steps)
        else:
            assert len(betas) == original_num_steps, 'betas must be the right length!'
            betas = np.array(betas, dtype=np.float64)

        # Rescale betas to match the number of rescaled diffusion steps with (eq. 19 in IDDPM)
        alphas = 1.0 - betas  # array of alpha_t for indices t
        alphas_cumprod = np.cumprod(alphas, axis=0)  # alphabar_t
        rescaled_timesteps = list(range(original_num_steps // (2 * rescaled_num_steps),
                                        original_num_steps + original_num_steps // (2 * rescaled_num_steps),
                                        original_num_steps // rescaled_num_steps))
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in rescaled_timesteps:
                new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        betas = np.array(new_betas)
        assert (betas > 0).all() and (betas <= 1).all(), 'betas in invalid range'

        self.betas = betas  # scheduled noise variance for each timestep
        self.timestep_map = torch.tensor(rescaled_timesteps, device=self.device,
                                         dtype=torch.long)  # map from rescaled to original timesteps

        # calculate and store various values to be used in diffusion, denoising, and ddim denoising
        # All these are arrays whose values at index t correspond to the comments next to them
        alphas = 1.0 - betas  # alpha_t
        sqrt_alphas = np.sqrt(alphas)  # sqrt(alpha_t)
        self.alphas_cumprod = alphas_cumprod = np.cumprod(alphas, axis=0)  # alphabar_t
        self.alphas_cumprod_prev = alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # alphabar_{t-1}
        self.sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)  # sqrt(alphabar_t)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)  # sqrt(1 - alphabar_t)
        self.sqrt_reciprocal_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)  # sqrt(1 / alphabar_t)
        self.sqrt_reciprocal_alphas_minus_one_cumprod = \
            np.sqrt(1.0 / alphas_cumprod - 1)  # sqrt((1 - alphabar_t) / alphabar_t)

        # Calculate posterior means and variances for forward (i.e. q sampling/diffusion) process, (eq. 7 in DDPM)
        self.posterior_mean_coef_x0 = np.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        self.posterior_mean_coef_xt = sqrt_alphas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Clip to remove variance at t=0, since it will be strange
        self.log_posterior_var_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))

    @torch.no_grad()
    def diffuse(self, x_0, steps_to_do=None, noise=None):
        """
        Add noise to an input corresponding to a given number of steps in the diffusion Markov chain.

            Parameters:
                - x (torch.tensor): input image to diffuse (usually x_0)
                - steps_to_do (int): number of rescaled diffusion steps to apply forward

            Returns:
                - Diffused image(s).
        """
        with torch.no_grad():
            # If unspecified or invalid number of steps to do, go until completion x_T
            if steps_to_do is None or steps_to_do > self.rescaled_num_steps:
                steps_to_do = self.rescaled_num_steps

            timestep = ((steps_to_do - 1) * torch.ones(x_0.shape[0])).to(self.device)
            x = self.diffusion_step(x_0, t=timestep, noise=noise)
        return x

    @torch.no_grad()
    def denoise(self, x=None, kwargs=None,
                start_step=None, steps_to_do=None, batch_size=1, ema_params=None, progress=True):
        """
        Sample the posterior of the forward process in the diffusion Markov chain. If self.use_ddim is True, uses DDIM
        sampling instead of traditional DDPM sampling.

            Parameters:
                - x (torch.tensor): if not None, input image x_t to denoise. if None, denoises Gaussian noise x_T
                - kwargs (dict): dict of extra args to pass to model, should be {'y': label}  with label to guide with
                - start_step (int): which rescaled step to start at. should correspond to x's timestep
                - steps_to_do (int): number of rescaled diffusion steps to apply forward
                - batch_size (int): batch size of denoising process. only used if x is None
                - ema_params (dict): dictionary of EMA parameters' data. if not None, use these for sampling
                - progress (bool): whether to show tqdm progress bar

            Returns:
                - Denoised sample(s).
        """
        if kwargs is None:
            kwargs = {}

        assert ('y' in kwargs.keys() and kwargs['y'] is not None) == self.model.conditional, \
            'pass label iff model is class-conditional'
        if self.model.conditional:
            assert len(kwargs['y']) == batch_size, 'len(labels) != batch size'
        with torch.no_grad():
            # Replace model parameters with ema parameters if desired
            if ema_params is not None:
                original_params = {}
                for name, param in self.model.named_parameters():
                    original_params[name] = param.data
                    param.data = ema_params[name].to(self.device)

            # If no specified starting step, start from x = x_T
            if start_step is None:
                start_step = self.rescaled_num_steps

            # If unspecified or invalid number of steps to do, go until completion x_0
            if steps_to_do is None or steps_to_do > start_step:
                steps_to_do = start_step

            # If unspecified x, start with x_T that is noise
            if x is None:
                assert start_step == self.rescaled_num_steps, 'cannot start from noise with current step that is not T'
                x = torch.randn(batch_size, self.model.in_channels, self.model.resolution, self.model.resolution). \
                    to(self.device)

            # Apply timestep rescaling
            indices = list(range(steps_to_do))
            # Add progress bar if needed
            if progress:
                progress_bar = tqdm.tqdm
                indices = progress_bar(reversed(indices), total=steps_to_do)
            else:
                indices = list(reversed(indices))

            assert len(indices) == steps_to_do
            for t in indices:
                timestep = (t * torch.ones(x.shape[0])).to(self.device)
                if not self.use_ddim:  # NORMAL DDPM
                    x, x_0 = self.denoising_step(x, t=timestep, kwargs=kwargs)
                else:  # DDIM SAMPLING
                    x, x_0 = self.ddim_denoising_step(x, t=timestep, kwargs=kwargs)

            # Reset model back to non-ema parameters
            if ema_params is not None:
                for name, param in self.model.named_parameters():
                    param.data = original_params[name]
        return x

    # -----------------------------------------------------------------------------------------------------------------
    # Calculations for each step
    # -----------------------------------------------------------------------------------------------------------------

    def diffusion_step(self, x_0, t, noise=None):
        """
        Samples from q(x_t | x_0), i.e. applies t steps of noise to x_0.
        """
        if noise is None:
            noise = torch.randn_like(x_0).to(self.device)
        # (eq. 4 in DDPM paper)
        return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise

    def get_eps_and_log_var(self, x_t, t, kwargs):
        """
        Returns predicted epsilon and predicted (or fixed) log variance from model.
        """
        eps_pred = self.model(x_t, self.timestep_map[t.long()], **kwargs)

        if self.sampling_var_type == VarType.LEARNED:
            eps_pred, log_var = torch.split(eps_pred, int(eps_pred.shape[1] / 2), dim=1)
        elif self.sampling_var_type == VarType.LEARNED_INTERPOLATION:
            assert self.log_posterior_var_clipped is not None and self.betas is not None
            eps_pred, log_var = torch.split(eps_pred, int(eps_pred.shape[1] / 2), dim=1)
            # (eq. 1 in OpenAI paper)
            min_log = extract(self.log_posterior_var_clipped, t, x_t.shape)
            max_log = extract(np.log(self.betas), t, x_t.shape)
            frac = (log_var + 1) / 2
            log_var = frac * max_log + (1 - frac) * min_log
        elif self.sampling_var_type == VarType.LARGE:
            log_var = extract(np.log(np.append(self.posterior_variance[1], self.betas[1:])), t, x_t.shape)
        elif self.sampling_var_type == VarType.SMALL:
            log_var = extract(np.log(np.maximum(self.posterior_variance, 1e-20)), t, x_t.shape)
        else:
            raise NotImplementedError(self.sampling_var_type)
        return eps_pred, log_var

    def denoising_step(self, x_t, t, kwargs=None, clip_x=True):
        """
        Samples from p(x_{t-1} | x_t), i.e. use model to predict noise, then samples from corresponding possible
        x_{t-1}'s.
        If return_x0, also returns the predicted initial image x_0.
        """
        eps_pred, log_var = self.get_eps_and_log_var(x_t, t, kwargs=kwargs)

        # If using classifier-free guidance, push eps_pred in the direction unique to its class and
        # away from the base prediction (eq. 6 in CFDG paper):
        # eps_pred(x_t, c) <- (1 + w) * eps_pred(x_t, c) - w * eps_pred(x_t, -1), where 0 is the null class
        if self.guidance == 'classifier_free':
            base_eps_pred = self.model(x_t, self.timestep_map[t.long()],
                                       y=torch.tensor([0] * eps_pred.shape[0], device=self.device))
            if self.sampling_var_type == VarType.LEARNED or self.sampling_var_type == VarType.LEARNED_INTERPOLATION:
                base_eps_pred, _ = torch.split(base_eps_pred, int(base_eps_pred.shape[1] / 2), dim=1)
            eps_pred = (1 + self.strength) * eps_pred - self.strength * base_eps_pred

        # Predict x_start from x_t and epsilon (eq. 11 in DDPM paper)
        pred_x0 = extract(self.sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_reciprocal_alphas_minus_one_cumprod, t, eps_pred.shape) * eps_pred
        if clip_x:
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Calculate mean of posterior q(x_{t-1} | x_t, x_0) (eq. 7 in DDPM paper)
        mean = extract(self.posterior_mean_coef_x0, t, pred_x0.shape) * pred_x0 + extract(
            self.posterior_mean_coef_xt, t, x_t.shape) * x_t

        # If we use classifier guidance, add to the mean the value: s * grad_{x_t}[log(classifier prob)]
        # This is (Algorithm 1 in OpenAI paper)
        if self.guidance == 'classifier':
            with torch.enable_grad():
                x = x_t.detach().requires_grad_(True)
                classifier_log_probs = torch.log_softmax(self.classifier(x, t), dim=-1)
                # Grab log probabilities of desired labels for each element of batch
                grabbed = classifier_log_probs[range(classifier_log_probs.shape[0]), torch.flatten(kwargs['y'])]
                grad = torch.autograd.grad(grabbed.sum(), x)[0]  # grad = grad_{x_t}[log(p[y | x_t, t])]
                mean += self.strength * grad * torch.exp(log_var)

        # Return sample pred for x_0 using calculated mean and given log variance, evaluated at desired timestep
        # (Between eq. 11 and eq. 12 in DDPM paper)
        noise = torch.randn_like(x_t)
        mask = 1.0 - (t == 0).float()
        mask = mask.reshape((x_t.shape[0],) + (1,) * (len(x_t.shape) - 1))

        sample = mean + mask * torch.exp(0.5 * log_var) * noise
        sample = sample.float()

        return sample, pred_x0

    def ddim_denoising_step(self, x_t, t, kwargs=None, clip_x=True):
        """
        Implement denoising diffusion implicit models (DDIM): https://arxiv.org/pdf/2010.02502.pdf
        """
        eps_pred = self.model(x_t, self.timestep_map[t.long()], **kwargs)
        if self.sampling_var_type == VarType.LEARNED or self.sampling_var_type == VarType.LEARNED_INTERPOLATION:
            eps_pred, _ = torch.split(eps_pred, int(eps_pred.shape[1] / 2), dim=1)

        # If we use classifier guidance, subtract from eps_pred the value:
        # s * sqrt(1 - alpha_bar) grad_{x_t}[log(classifier prob)]
        # This is (Algorithm 2 and eq. 14 in OpenAI paper)
        if self.guidance == 'classifier':
            with torch.enable_grad():
                x = x_t.detach().requires_grad_(True)
                classifier_log_probs = torch.log_softmax(self.classifier(x, t), dim=-1)
                # Grab log probabilities of desired labels for each element of batch
                grabbed = classifier_log_probs[range(classifier_log_probs.shape[0]), torch.flatten(kwargs['y'])]
                grad = torch.autograd.grad(grabbed.sum(), x)[0]  # grad = grad_{x_t}[log(p[y | x_t, t])]
                eps_pred -= self.strength * grad * extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        # If using classifier-free guidance, push eps_pred in the direction unique to its class and
        # away from the base prediction (eq. 6 in CFDG paper):
        # eps_pred(x_t, c) <- (1 + w) * eps_pred(x_t, c) - w * eps_pred(x_t, -1), where 0 is the null class
        elif self.guidance == 'classifier_free':
            base_eps_pred = self.model(x_t, self.timestep_map[t.long()],
                                       y=torch.tensor([0] * eps_pred.shape[0], device=self.device))
            if self.sampling_var_type == VarType.LEARNED or self.sampling_var_type == VarType.LEARNED_INTERPOLATION:
                base_eps_pred, _ = torch.split(base_eps_pred, int(base_eps_pred.shape[1] / 2), dim=1)
            eps_pred = (1 + self.strength) * eps_pred - self.strength * base_eps_pred

        # Same as in DDPM (eq. 11)
        pred_x0 = extract(self.sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_reciprocal_alphas_minus_one_cumprod, t, x_t.shape) * eps_pred
        if clip_x:
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Accelerated sampling of Generative Process (secs. 4.1 and 4.2 in DDIM)
        # (eq. 12 in DDIM)
        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x_t.shape)
        var = self.ddim_eta ** 2 * (1.0 - alpha_bar_prev) * (1.0 - alpha_bar / alpha_bar_prev) / (1.0 - alpha_bar)
        mean = pred_x0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - var) * eps_pred

        noise = torch.randn_like(x_t)
        mask = 1.0 - (t == 0).float()
        mask = mask.reshape((x_t.shape[0],) + (1,) * (len(x_t.shape) - 1))

        sample = mean + mask * torch.sqrt(var) * noise
        sample = sample.float()

        return sample, pred_x0

    # -----------------------------------------------------------------------------------------------------------------
    # Loss calculations
    # -----------------------------------------------------------------------------------------------------------------

    def loss(self, x_0, t, kwargs=None, noise=None):
        """
        Returns training loss in units of bits per dimension for training step involving batch x_0 and timestep t
        (and possible labels kwargs['y']), where:
            We diffuse x_0 to x_t, predict epsilon from x_t, and calculate loss compared to the applied noise either
            using MSE directly ('simple' loss), using KL-divergence on the resulting distributions of samples
            ('KL' loss), or a weighted sum of the two ('hybrid' loss).
        x_0 and t (and possibly kwargs['y'] and noise) are expected to be tensors.
        """
        if kwargs is None:
            kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_0).to(self.device)

        # Add t steps of noise to get x_t from x_0
        x_t = self.diffusion_step(x_0=x_0, t=t, noise=noise)

        # Get model outputs (eps_pred and either predicted or fixed log_var)
        eps_pred, log_var = self.get_eps_and_log_var(x_t, t, kwargs=kwargs)

        # Simple loss is MSE between predicted and actual noise (eq. 14 of DDPM)
        if self.loss_type == LossType.SIMPLE:
            loss = mean_flat((eps_pred - noise) ** 2)
        # KL loss is KL-Divergence of predicted and actual Gaussian distributions
        elif self.loss_type == LossType.KL or self.loss_type == LossType.KL_RESCALED:
            loss = self.variational_lower_bound(x_0, x_t, t, eps_pred, log_var)
            if self.loss_type == LossType.KL_RESCALED:
                loss *= self.rescaled_num_steps
        # Hybrid loss (eq. 16 of IDDPM paper) uses weighted sum of simple loss from (eq. 14 of DDPM)
        # and KL-Divergence loss (with frozen gradients from eps_pred, only allowing updates coming from log_var)
        else:
            loss_simple = mean_flat((eps_pred - noise) ** 2)
            eps_pred_detached = eps_pred.detach()  # ensure that loss_vlb doesn't backprop through eps_pred, only var
            loss_vlb = self.variational_lower_bound(x_0, x_t, t, eps_pred_detached, log_var) * self.rescaled_num_steps
            loss = loss_simple + 0.001 * loss_vlb
        return loss

    def variational_lower_bound(self, x_0, x_t, t, eps_pred, log_var):
        """
        Returns variational lower bound term in units of bits per dimension.
        The true variational lower bound is the sum of these terms for all t.
        """
        # Get true mean and log_var of sampling distribution
        true_mean = extract(self.posterior_mean_coef_x0, t, x_0.shape) * x_0 + extract(
            self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        true_log_var = extract(self.log_posterior_var_clipped, t, x_0.shape)

        pred_x0 = extract(self.sqrt_reciprocal_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_reciprocal_alphas_minus_one_cumprod, t, eps_pred.shape) * eps_pred

        # Calculate model-predicted mean of posterior q(x_{t-1} | x_t, x_0) (eq. 7 in DDPM paper)
        mean = extract(self.posterior_mean_coef_x0, t, pred_x0.shape) * pred_x0 + extract(
            self.posterior_mean_coef_xt, t, x_t.shape) * x_t

        # Calculate KL-Divergence between N(true_mean, true_log_var) and N(mean, log_var) to find output
        kl = kl_div(true_mean, true_log_var, mean, log_var)
        kl = mean_flat(kl) / np.log(2.0)  # convert from nats to bits

        # Calculate negative log-likelihood of start image appearing in predicted distribution
        nll = -log_likelihood(x_0, mean, log_var)
        nll = mean_flat(nll) / np.log(2.0)  # convert from nats to bits

        # At the first timestep return the nll, otherwise return KL(q(x_{t-1}|x_t,x_0), p(x_{t-1}|x_t))
        return torch.where((t == 0), nll, kl)


# ---------------------------------------------------------------------------------------------------------------------
# Helper methods!
# ---------------------------------------------------------------------------------------------------------------------

def get_beta_schedule(schedule_method, num_steps, beta_0, beta_T):
    """
    Returns schedule for desired noise variance (beta) at each timestep.

        Parameters:
            - schedule_method (str): method to use for scheduling - either 'linear', 'constant', or 'cosine'
            - num_steps (int): number of diffusion steps to create betas for (should be T)
            - beta_0 (double): value to use for initial beta in linear method and constant beta in constant method
            - beta_T (double): value to use for final beta in linear method
        Returns:
            - betas (np.array): array of size (num_steps,) of beta values to use at each step
    """
    if schedule_method == 'linear':
        betas = np.linspace(beta_0, beta_T, num_steps, dtype=np.float64)
    elif schedule_method == 'constant':
        betas = beta_0 * np.ones(num_steps, dtype=np.float64)
    elif schedule_method == 'cosine':  # from (eq. 17 of IDDPM)
        # function f(t) described in (eq. 17 of IDDPM)
        def f(t):
            s = 0.008  # extra value to add to fraction to prevent singularity
            return math.cos((t + s) / (1.0 + s) * math.pi / 2) ** 2

        betas = []
        for step in range(num_steps):
            alphabar_t_minus_one = step / num_steps
            alphabar_t = (step + 1) / num_steps
            betas.append(min(1 - f(alphabar_t) / f(alphabar_t_minus_one), 0.999))  # clip beta to be <= 0.999
        return np.array(betas)
    else:
        raise NotImplementedError("unimplemented variance scheduling method: {}".format(schedule_method))
    return betas


def extract(a, t, broadcast_shape):
    """
    Index into a's elements with index t, returning tensor that is broadcastable to shape.

            Parameters:
            - a (np.array): array to index into
            - t (torch.tensor): timestep tensor to use as index
            - broadcast_shape (iterable): shape to make output tensor broadcastable to
        Returns:
            - result (torch.tensor): tensor of selected values from a, broadcastable with broadcast_shape
    """
    result = torch.gather(torch.from_numpy(a).to(t.device).float(), 0, t.long())
    # Keep adding dimensions to results until right shape
    while len(result.shape) < len(broadcast_shape):
        result = result[..., None]
    return result.expand(broadcast_shape)


def kl_div(mean_1, log_var_1, mean_2, log_var_2):
    """
    Returns KL-Divergence between two Gaussian's with given parameters.
    Expects tensor inputs and returns tensor output for use in loss functions.
    Output is measured in nats (log base e).
    """
    # Formula is KL(p,q) = log(sigma2/sigma1) + (sigma1^2+(mu1-mu2)^2)/(2sigma2^2) - 1/2
    #                    = log(var2/var1)/2 + var1/2var2 + (mu1-mu2)^2/2sigma2^2 - 1/2
    return ((log_var_2 - log_var_1) + torch.exp(log_var_1 - log_var_2) +
            ((mean_1 - mean_2) ** 2) * torch.exp(-log_var_2) - 1.0) / 2


def approx_cdf(x):
    """
    Returns approximate value of cdf(x) for the Gaussian with zero mean and unit variance. x can be a tensor.
    Approximation is from:
    "Page, E. (1977). Approximations to the cumulative normal function and its inverse for use on a pocket calculator."
    """
    y = math.sqrt(2.0 / math.pi) * (x + 0.0444715 * (x ** 3))
    return 0.5 * (1.0 + torch.tanh(y))


def log_likelihood(target, mean, log_var):
    """
    Returns log-likelihood of Gaussian with given mean and log variance, discretized to target image.
    Expects tensor inputs and returns tensor output for use in loss functions.
    target should be an image with values in [-1.0, 1.0]
    Output is measured in nats (log base e).
    """
    assert target.shape == mean.shape == log_var.shape
    std_recip = torch.exp(-0.5 * log_var)
    centered = target - mean

    # calculate z-scores of discretized steps to target image
    plus = (centered + 1.0 / 255.0) * std_recip
    minus = (centered - 1.0 / 255.0) * std_recip
    cdf_minus, cdf_plus = approx_cdf(minus), approx_cdf(plus)
    cdf_delta = cdf_plus - cdf_minus

    # calculate log probabilities (we ensure that cdfs are not smaller than 1e-12 before taking log)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_minus = torch.log((1.0 - cdf_minus).clamp(min=1e-12))
    return torch.where(target < -0.999, log_cdf_plus,
                       torch.where(target > 0.999, log_one_minus_cdf_minus, torch.log(cdf_delta.clamp(min=1e-12))))


def mean_flat(tensor):
    """
    Takes mean over all the non-batch dimensions of the tensor.
    """
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


class VarType(enum.Enum):
    """
    Enum to keep track of how we learn sampling variances.
    """
    SMALL = enum.auto()  # Use small fixed variances
    LARGE = enum.auto()  # Use large fixed variances
    LEARNED = enum.auto()  # Learn covariance directly
    LEARNED_INTERPOLATION = enum.auto()  # Learn covariance by learning value to use to interpolate between min and max

    @staticmethod
    def get_var_type(sampling_var_type):
        if sampling_var_type == 'small':
            return VarType.SMALL
        elif sampling_var_type == 'large':
            return VarType.LARGE
        elif sampling_var_type == 'learned':
            return VarType.LEARNED
        elif sampling_var_type == 'learned_interpolation':
            return VarType.LEARNED_INTERPOLATION
        else:
            raise NotImplementedError(sampling_var_type)


class LossType(enum.Enum):
    """
    Enum to keep track of method we use for calculating loss.
    """
    SIMPLE = enum.auto()  # use raw MSE loss between eps_pred and applied noise
    KL = enum.auto()  # use the variational lower-bound
    KL_RESCALED = enum.auto()  # like KL, but rescale to estimate the full VLB
    HYBRID = enum.auto()  # use weighted sum of SIMPLE for learning eps_pred and KL for learning variances

    @staticmethod
    def get_loss_type(loss_type):
        if loss_type == 'simple':
            return LossType.SIMPLE
        elif loss_type == 'KL':
            return LossType.KL
        elif loss_type == 'KL_rescaled':
            return LossType.KL_RESCALED
        elif loss_type == 'hybrid':
            return LossType.HYBRID
        else:
            raise NotImplementedError(loss_type)
