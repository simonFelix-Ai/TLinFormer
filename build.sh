####################################################################################
# Copyright (c) 2025, Zhongpan Tang
#
# Licensed under the Academic and Non-Commercial Research License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   LICENSE.md file in the repository
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For commercial use, please contact: tangzhongp@gmail.com
####################################################################################

TOPDIR=$(cd `dirname $0`; pwd)

RESULTDIR=$TOPDIR/results

HARD_INFO_FILE=$RESULTDIR/hardware.info

collect_hardware_info()
{
    cd $TOPDIR

    set -x

    nvidia-smi 
    nvidia-smi topo -m

    python -c "import torch; print(torch.__version__)"
    python -c "import torch; print(torch.version.cuda)"
    python -c "import torch.backends.cudnn; print(torch.backends.cudnn.version())"

    nvcc --version  

    python --version 

    lscpu  

    free -h
    
    echo =========== cat /proc/meminfo start ================= 
    cat /proc/meminfo  

    lsblk  

    lsb_release -a  

    cat /etc/os-release  

    uname -a  
 
    pip list 

    df -h

    set +x
}

train_all()
{
    cd $TOPDIR

    for config in $(ls configs); do
        echo use $config
        python train.py --config configs/$config
    done
}

mkdir -p $RESULTDIR
collect_hardware_info >> $HARD_INFO_FILE 2>&1
train_all
