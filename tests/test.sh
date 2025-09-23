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

ProjectDIR=$(cd `dirname $0`; pwd)/../

do_test()
{
    cd $ProjectDIR

    configs="$(ls configs)"
    test_cases="$(basename -s .py $ProjectDIR/tests/test_*)"

    cd $ProjectDIR

    for config in $configs; do
        for test_case in $test_cases; do
            set -x
            python -m tests.$test_case --config $config
            set +x
        done
    done
}

do_test
