from setuptools import setup
from Cython.Build import cythonize

setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
        'crowd_sim.envs.env_manager',
        'crowd_sim.envs.env_manager.base_managers',
    ],
    install_requires=[
        'gitpython',
        'pandas==2.0.3',
        'gym==0.23.1',
        'matplotlib',
        'numpy==1.23.4',
        'scipy',
        'gymnasium==0.29.0',
        'torch==2.2.2',
        'stable-baselines3==2.1.0',
        'sb3-contrib==2.1.0',
        'imitation==1.0.0',
        'torchvision',
        'seaborn==0.8.1',
        'tqdm==4.65.0',
        'tensorboardX',
        'opencv-python==4.9.0',
        'numba==0.58.1',
        'omegaconf==2.3.0',
        'hydra==1.2.0',
        'jaxlib==0.4.13',
        'jax==0.4.13',
        'flax==0.6.9',

    ],
    ext_modules=cythonize("crings/crings.pyx", annotate=True),
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
