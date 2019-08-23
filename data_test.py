# test integrity of the input data
"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import os
import numpy as np
import pandas as pd
from azure.storage.blob import BlockBlobService


STORAGEACCOUNTNAME="gepstorageaccsample"
STORAGEACCOUNTKEY="LiY3PTqRinUDvrIWJGXCP8cnGZ1vr4QbVSDIHgOZRjvxEW9JhnjSuK8JdQdIo411L9O5eXWgaORHThChLrtw3w=="
CONTAINERNAME="gepcontainer"

def load_file(filename):
    blob_service=BlockBlobService(account_name=STORAGEACCOUNTNAME,account_key=STORAGEACCOUNTKEY)
    blob_service.get_blob_to_path(CONTAINERNAME,"input/"+filename,filename)


def save_file(filename):
    blob_service=BlockBlobService(account_name=STORAGEACCOUNTNAME,account_key=STORAGEACCOUNTKEY)
    blob_service.create_blob_from_path(CONTAINERNAME,"output/"+filename,filename)


# number of features
expected_columns = 10

# distribution of features in the training set
historical_mean = np.array(
    [
        -3.63962254e-16,
        1.26972339e-16,
        -8.01646331e-16,
        1.28856202e-16,
        -8.99230414e-17,
        1.29609747e-16,
        -4.56397112e-16,
        3.87573332e-16,
        -3.84559152e-16,
        -3.39848813e-16,
        1.52133484e02,
    ]
)
historical_std = np.array(
    [
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        4.75651494e-02,
        7.70057459e01,
    ]
)

# maximal relative change in feature mean or standrd deviation
# that we can tolerate
shift_tolerance = 3


def test_check_schema():
    filename="diabetes.csv"
    load_file(filename)
    # check that file exists
    assert os.path.exists(filename)
    dataset = pd.read_csv(filename)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns == expected_columns
    save_file(filename)


def test_check_bad_schema():
    filename="diabetes_bad_schema.csv" 
    load_file(filename)
    # check that file exists
    assert os.path.exists(filename)
    dataset = pd.read_csv(filename)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns != expected_columns
    save_file(filename)


def test_check_missing_values():
    filename="diabetes_missing_values.csv" 
    load_file(filename)
    # check that file exists
    assert os.path.exists(filename)
    dataset = pd.read_csv(filename)
    n_nan = np.sum(np.isnan(dataset.values))
    assert n_nan > 0
    save_file(filename)


def test_check_distribution():
    filename="diabetes_bad_dist.csv" 
    load_file(filename)
    # check that file exists
    assert os.path.exists(filename)
    dataset = pd.read_csv(filename)
    mean = np.mean(dataset.values, axis=0)
    std = np.mean(dataset.values, axis=0)
    assert (
        np.sum(abs(mean - historical_mean) > shift_tolerance * abs(historical_mean))
        or np.sum(abs(std - historical_std) > shift_tolerance * abs(historical_std)) > 0
    )
    save_file(filename)


test_check_schema()
test_check_bad_schema()
test_check_missing_values()
test_check_distribution()