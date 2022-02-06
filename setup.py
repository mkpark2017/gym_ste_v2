from setuptools import setup

setup(
    name='gym_ste_v2',
    version='0.2.0',
    url='https://github.com/mkpark2017/gym_ste_v2',
    Author='Minkyu Park',
    packages=['gym_ste_v2'],
    install_requires=['gym>=0.18.3','numpy>=1.21.0', 'ipdb'],
    description='OpenAI GYM environments for source term estimation and sample learning codes',
)
