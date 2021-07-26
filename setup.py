from setuptools import setup

setup(
    name='gym_ste',
    version='0.0.3',
    url='https://github.com/mkpark2017/gym-ste',
    Author='Minkyu Park',
    packages=['gym_ste'],
    install_requires=['gym>=0.18.3','numpy>=1.21.0', 'ipdb'],
    description='OpenAI GYM environments for source term estimation and sample learning codes',
)
