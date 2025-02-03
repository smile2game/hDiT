from setuptools import setup, find_packages

setup(
    name='hfuser',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # 在这里列出你的依赖包
        'numpy',  # 举例
    ],
    # 你可以根据需要添加更多元数据
    author='smile2game',
    author_email='2426827419@qq.com',
    description='hfuser for DiT parallelism',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
