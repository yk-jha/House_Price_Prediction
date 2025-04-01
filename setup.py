from setuptools import find_packages,setup

from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->list[str]:
    '''this function return requirements'''
    reqd = []
    with open(file_path) as file_obj:
        reqd = file_obj.readlines()
        reqd=[req.replace("\n" , "") for req in reqd ]
        
        if HYPHEN_E_DOT in reqd:
            reqd.remove(HYPHEN_E_DOT)
            
    return reqd



setup(
    name="HousePriceprediction",
    version='0.0.1',
    author='Yash',
    author_email='yjha5965@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
