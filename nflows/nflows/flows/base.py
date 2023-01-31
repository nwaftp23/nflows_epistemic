"""Basic definitions for the flows module."""


import torch.nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context, kwargs=None):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context, kwargs=kwargs)
        log_prob = self._distribution.log_prob(noise, context=embedded_context, kwargs=kwargs)
        return log_prob + logabsdet

    def log_prob_base(self, inputs, context, kwargs=None):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context, kwargs=kwargs)
        log_prob = self._distribution.log_prob(noise, context=embedded_context, kwargs=kwargs)
        return log_prob + logabsdet, log_prob
    
    def _sample(self, num_samples, context, kwargs=None):
        embedded_context = self._embedding_net(context)
        
        noise, means, stds = self._distribution.sample(num_samples, 
                context=embedded_context, kwargs=kwargs)
        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context, kwargs=kwargs)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            noise = torchutils.split_leading_dim(noise, shape=[-1, num_samples])

        return samples, noise, means, stds 

    def smp_lp(self, num_samples, context=None, kwargs=None):
        embedded_context = self._embedding_net(context)
        noise, log_prob, means, stds = self._distribution.sample_and_log_prob(
            num_samples, context=embedded_context, kwargs=kwargs
        )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)
        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])
            noise = torchutils.split_leading_dim(noise, shape=[-1, num_samples])
    
        return samples, log_prob - logabsdet, log_prob, noise, means, stds


    def sample_and_log_prob(self, num_samples, context=None, kwargs=None, 
        ensemble=False, ensemble_size=1):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        kwargs_cp = kwargs.copy()
        samples, nflows_log_prob, base_log_prob, base_out, base_means, base_stds =(
            self.smp_lp(num_samples, context=context, kwargs=kwargs_cp))
        nflows_comp_log_prob = nflows_log_prob
        if ensemble:
            out_probs = []
            base_probs = []
            if num_samples > 1:
                context =  context.repeat_interleave(num_samples, dim=0)
            for i in range(ensemble_size):
                kwargs_cp['mask_index'] = i
                out_log_prob, base_log_prob = self.log_prob_base(
                        samples.reshape(samples.shape[0]*samples.shape[1],-1), 
                        context=context, kwargs=kwargs_cp)
                out_probs.append(torch.exp(out_log_prob))
                base_probs.append(torch.exp(base_log_prob))
            out_log_probs = torch.log((torch.stack(out_probs)).mean(0))
            base_log_prob = torch.log((torch.stack(base_probs)).mean(0))
            nflows_log_prob = out_log_probs.reshape(nflows_log_prob.shape)
            base_log_prob = base_log_prob.reshape(nflows_log_prob.shape) 
        return (samples, nflows_log_prob, nflows_comp_log_prob, 
            base_log_prob, base_out, base_means, base_stds)

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise
