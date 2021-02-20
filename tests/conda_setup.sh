#/bin/bash

if [[ "$DISTRIB" == "conda" ]]; then	
    # conda-based environment instead	
    deactivate

    if [[ -f "$HOME/miniconda/bin/conda" ]]; then	
        echo "Skip install conda [cached]"	
    else	
        # By default, travis caching mechanism creates an empty dir in the	
        # beginning of the build, but conda installer aborts if it finds an	
        # existing folder, so let's just remove it:	
        rm -rf "$HOME/miniconda"	

        # Use the miniconda installer for faster download / install of conda	
        # itself	
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \	
            -O miniconda.sh	
        chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda	
    fi	
    export PATH=$HOME/miniconda/bin:$PATH	
    # Make sure to use the most updated version	
    conda update --yes conda	

    # Configure the conda environment and put it in the path using the	
    # provided versions	
    # (prefer local venv, since the miniconda folder is cached)	
    # conda create -p ./.venv --yes python=${PYTHON_VERSION} pip virtualenv	
    # source activate ./.venv
fi

if [[ -f "/opt/conda/bin/python3.6" ]]; then	
    echo "Skip install python 3.6"
else	
    conda create -y -n tox-py36 python=3.6
    sudo ln -s /opt/conda/envs/tox-py36/bin/python3.6 /opt/conda/bin/python3.6
fi

if [[ -f "/opt/conda/bin/python3.7" ]]; then	
    echo "Skip install python 3.7"
else	
    conda create -y -n tox-py37 python=3.7
    sudo ln -s /opt/conda/envs/tox-py37/bin/python3.7 /opt/conda/bin/python3.7
fi

if [[ -f "/opt/conda/bin/python3.8" ]]; then	
    echo "Skip install python 3.8"
else	
    conda create -y -n tox-py38 python=3.8
    sudo ln -s /opt/conda/envs/tox-py38/bin/python3.8 /opt/conda/bin/python3.8
fi

