# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch as th
import numpy as np


# 计算两个高斯分布之间的KL散度。
# KL散度是衡量两个概率分布之间差异的方法。
# 在VAE和生成模型中，KL散度被用来衡量真实分布和近似分布之间的差异。
# 使用：
# （1）这个函数可能用于计算潜在空间中的编码分布与先验分布之间的差异，这是VAEs中常用的技术。通过最小化这个散度，DiT模型可以学习到更好的数据表示。
# （2）这个函数可能在模型的损失函数中用于计算VAE的部分，或者在训练过程中用于评估模型学习到的分布与目标分布之间的差异
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    mean1, logvar1：第一个高斯分布的均值和方差；
    mean2, logvar2：第二个高斯分布的均值和方差；
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


# 用于计算标准正态分布的累积分布函数（CDF）的快速近似值。累积分布函数给出了随机变量小于或等于某个值的概率。
# 在生成模型中，常常需要从标准正态分布中采样。通过计算逆CDF，可以在给定的概率区间内生成对应的随机变量值。
# 在训练过程中，可能需要计算数据或模型输出与标准正态分布之间的匹配程度，这时会用到CDF来衡量概率。
# 在模型训练完成后，可能需要评估模型生成数据的分布与标准正态分布的接近程度，CDF可以用来进行这一评估。
def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


# 用于计算连续高斯分布的对数似然值
def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = th.distributions.Normal(th.zeros_like(x), th.ones_like(x)).log_prob(normalized_x)
    return log_probs


# 用于计算一个高斯分布被离散化到一个特定图像上的对数似然值。
# 在图像生成任务中，生成的图像通常是由像素值组成的，而这些像素值是离散的，例如，在0到255之间的整数。
# 因此，为了计算这些离散值相对于连续高斯分布的对数似然，需要进行离散化处理。
def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
