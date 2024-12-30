{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "f5e3dc56",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "47f01a6a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1.2</td>\n",
       "      <td>39344.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1.4</td>\n",
       "      <td>46206.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>1.6</td>\n",
       "      <td>37732.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>2.1</td>\n",
       "      <td>43526.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>2.3</td>\n",
       "      <td>39892.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>56643.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>60151.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>7</td>\n",
       "      <td>3.3</td>\n",
       "      <td>54446.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8</td>\n",
       "      <td>3.3</td>\n",
       "      <td>64446.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>9</td>\n",
       "      <td>3.8</td>\n",
       "      <td>57190.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>10</td>\n",
       "      <td>4.0</td>\n",
       "      <td>63219.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>11</td>\n",
       "      <td>4.1</td>\n",
       "      <td>55795.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>12</td>\n",
       "      <td>4.1</td>\n",
       "      <td>56958.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>13</td>\n",
       "      <td>4.2</td>\n",
       "      <td>57082.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>14</td>\n",
       "      <td>4.6</td>\n",
       "      <td>61112.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>15</td>\n",
       "      <td>5.0</td>\n",
       "      <td>67939.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>16</td>\n",
       "      <td>5.2</td>\n",
       "      <td>66030.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>17</td>\n",
       "      <td>5.4</td>\n",
       "      <td>83089.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>18</td>\n",
       "      <td>6.0</td>\n",
       "      <td>81364.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>19</td>\n",
       "      <td>6.1</td>\n",
       "      <td>93941.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>20</td>\n",
       "      <td>6.9</td>\n",
       "      <td>91739.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>21</td>\n",
       "      <td>7.2</td>\n",
       "      <td>98274.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>22</td>\n",
       "      <td>8.0</td>\n",
       "      <td>101303.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>23</td>\n",
       "      <td>8.3</td>\n",
       "      <td>113813.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>24</td>\n",
       "      <td>8.8</td>\n",
       "      <td>109432.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>25</td>\n",
       "      <td>9.1</td>\n",
       "      <td>105583.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>26</td>\n",
       "      <td>9.6</td>\n",
       "      <td>116970.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>27</td>\n",
       "      <td>9.7</td>\n",
       "      <td>112636.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>28</td>\n",
       "      <td>10.4</td>\n",
       "      <td>122392.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>29</td>\n",
       "      <td>10.6</td>\n",
       "      <td>121873.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Unnamed: 0  YearsExperience    Salary\n",
       "0            0              1.2   39344.0\n",
       "1            1              1.4   46206.0\n",
       "2            2              1.6   37732.0\n",
       "3            3              2.1   43526.0\n",
       "4            4              2.3   39892.0\n",
       "5            5              3.0   56643.0\n",
       "6            6              3.1   60151.0\n",
       "7            7              3.3   54446.0\n",
       "8            8              3.3   64446.0\n",
       "9            9              3.8   57190.0\n",
       "10          10              4.0   63219.0\n",
       "11          11              4.1   55795.0\n",
       "12          12              4.1   56958.0\n",
       "13          13              4.2   57082.0\n",
       "14          14              4.6   61112.0\n",
       "15          15              5.0   67939.0\n",
       "16          16              5.2   66030.0\n",
       "17          17              5.4   83089.0\n",
       "18          18              6.0   81364.0\n",
       "19          19              6.1   93941.0\n",
       "20          20              6.9   91739.0\n",
       "21          21              7.2   98274.0\n",
       "22          22              8.0  101303.0\n",
       "23          23              8.3  113813.0\n",
       "24          24              8.8  109432.0\n",
       "25          25              9.1  105583.0\n",
       "26          26              9.6  116970.0\n",
       "27          27              9.7  112636.0\n",
       "28          28             10.4  122392.0\n",
       "29          29             10.6  121873.0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salary=pd.read_csv('Salary_dataset.csv')\n",
    "salary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ef73f445",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>YearsExperience</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>30.000000</td>\n",
       "      <td>30.000000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>14.500000</td>\n",
       "      <td>5.413333</td>\n",
       "      <td>76004.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>8.803408</td>\n",
       "      <td>2.837888</td>\n",
       "      <td>27414.429785</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.200000</td>\n",
       "      <td>37732.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>7.250000</td>\n",
       "      <td>3.300000</td>\n",
       "      <td>56721.750000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>14.500000</td>\n",
       "      <td>4.800000</td>\n",
       "      <td>65238.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>21.750000</td>\n",
       "      <td>7.800000</td>\n",
       "      <td>100545.750000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>29.000000</td>\n",
       "      <td>10.600000</td>\n",
       "      <td>122392.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Unnamed: 0  YearsExperience         Salary\n",
       "count   30.000000        30.000000      30.000000\n",
       "mean    14.500000         5.413333   76004.000000\n",
       "std      8.803408         2.837888   27414.429785\n",
       "min      0.000000         1.200000   37732.000000\n",
       "25%      7.250000         3.300000   56721.750000\n",
       "50%     14.500000         4.800000   65238.000000\n",
       "75%     21.750000         7.800000  100545.750000\n",
       "max     29.000000        10.600000  122392.000000"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "salary.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a10009fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "salary=salary.drop('Unnamed: 0',axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "694f5d75",
   "metadata": {},
   "source": [
    "# feature extraction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ce7c47f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 30 entries, 0 to 29\n",
      "Data columns (total 2 columns):\n",
      " #   Column           Non-Null Count  Dtype  \n",
      "---  ------           --------------  -----  \n",
      " 0   YearsExperience  30 non-null     float64\n",
      " 1   Salary           30 non-null     float64\n",
      "dtypes: float64(2)\n",
      "memory usage: 608.0 bytes\n"
     ]
    }
   ],
   "source": [
    "salary.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "d40f842c",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=np.array(salary['YearsExperience']).reshape(-1,1)    #independent feature\n",
    "y=salary['Salary'].values              #dependent features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50974d4b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "aa09ccdc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.2],\n",
       "       [ 1.4],\n",
       "       [ 1.6],\n",
       "       [ 2.1],\n",
       "       [ 2.3],\n",
       "       [ 3. ],\n",
       "       [ 3.1],\n",
       "       [ 3.3],\n",
       "       [ 3.3],\n",
       "       [ 3.8],\n",
       "       [ 4. ],\n",
       "       [ 4.1],\n",
       "       [ 4.1],\n",
       "       [ 4.2],\n",
       "       [ 4.6],\n",
       "       [ 5. ],\n",
       "       [ 5.2],\n",
       "       [ 5.4],\n",
       "       [ 6. ],\n",
       "       [ 6.1],\n",
       "       [ 6.9],\n",
       "       [ 7.2],\n",
       "       [ 8. ],\n",
       "       [ 8.3],\n",
       "       [ 8.8],\n",
       "       [ 9.1],\n",
       "       [ 9.6],\n",
       "       [ 9.7],\n",
       "       [10.4],\n",
       "       [10.6]])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "2bd88594",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 39344.,  46206.,  37732.,  43526.,  39892.,  56643.,  60151.,\n",
       "        54446.,  64446.,  57190.,  63219.,  55795.,  56958.,  57082.,\n",
       "        61112.,  67939.,  66030.,  83089.,  81364.,  93941.,  91739.,\n",
       "        98274., 101303., 113813., 109432., 105583., 116970., 112636.,\n",
       "       122392., 121873.])"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "87b2bc60",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(30, 1) (30,)\n"
     ]
    }
   ],
   "source": [
    "print(x.shape,y.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02edf2a3",
   "metadata": {},
   "source": [
    "# to remove outlier we use normalization of data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e92da5ed",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "227a0747",
   "metadata": {},
   "source": [
    "# splitting into training and testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "e9c84617",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1.2.1'"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import sklearn\n",
    "sklearn.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "68ee196a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "f027b9b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,train_size=0.7)#by default it takes 70 percente testsize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "60bf2301",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(21, 1) (9, 1)\n",
      "(21,) (9,)\n"
     ]
    }
   ],
   "source": [
    "print(x_train.shape,x_test.shape)\n",
    "print(y_train.shape,y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "e299e205",
   "metadata": {},
   "outputs": [],
   "source": [
    "# model building\n",
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "21d4bb19",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr_re=LinearRegression()\n",
    "lr_re.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "ed992f19",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 55999.25471534,  73641.01158825,  75498.03862751, 109853.0388537 ,\n",
       "        44857.09247982, 121923.71460885,  92211.28198079, 102424.93069669,\n",
       "        62498.84935273])"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred=lr_re.predict(x_test)\n",
    "pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "0917a849",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 54446.,  66030.,  83089., 105583.,  43526., 122392.,  98274.,\n",
       "       113813.,  63219.])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "b6c1b3d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9438971895077892"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import r2_score\n",
    "r2_score(pred,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "8e2fdf9f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(4,4))\n",
    "plt.scatter(x_train,y_train,label='Actual')\n",
    "plt.title('Linear Regression')\n",
    "plt.xlabel('Experience')\n",
    "plt.ylabel('Salary')\n",
    "plt.plot(x_test,pred,color='hotpink',label='Predicted')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "e3f0c84e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[3.4]]\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "exp=np.array([[3.4]])\n",
    "print(exp)\n",
    "print(exp.ndim)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "2a5a8b66",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_o = lr_re.predict(exp)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "f8b61872",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([56927.76823497])"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_o"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "2444ea11",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[5.]]\n",
      "2\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([71783.984549])"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "exp=np.array([[5.0]])\n",
    "print(exp)\n",
    "print(exp.ndim)\n",
    "new_o = lr_re.predict(exp)\n",
    "new_o"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24fd18a9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
