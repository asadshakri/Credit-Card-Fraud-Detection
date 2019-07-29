{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import keras\n",
    "\n",
    "np.random.seed(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv('creditcard.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## data exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
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
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Class</th>\n",
       "      <th>normalizedAmount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>0</td>\n",
       "      <td>0.244964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.342475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>...</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>0</td>\n",
       "      <td>1.160686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>0</td>\n",
       "      <td>0.140534</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.073403</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9        ...              V21       V22       V23  \\\n",
       "0  0.098698  0.363787        ...        -0.018307  0.277838 -0.110474   \n",
       "1  0.085102 -0.255425        ...        -0.225775 -0.638672  0.101288   \n",
       "2  0.247676 -1.514654        ...         0.247998  0.771679  0.909412   \n",
       "3  0.377436 -1.387024        ...        -0.108300  0.005274 -0.190321   \n",
       "4 -0.270533  0.817739        ...        -0.009431  0.798278 -0.137458   \n",
       "\n",
       "        V24       V25       V26       V27       V28  Class  normalizedAmount  \n",
       "0  0.066928  0.128539 -0.189115  0.133558 -0.021053      0          0.244964  \n",
       "1 -0.339846  0.167170  0.125895 -0.008983  0.014724      0         -0.342475  \n",
       "2 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752      0          1.160686  \n",
       "3 -1.175575  0.647376 -0.221929  0.062723  0.061458      0          0.140534  \n",
       "4  0.141267 -0.206010  0.502292  0.219422  0.215153      0         -0.073403  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## data pre-processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'Amount'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3077\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3078\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3079\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'Amount'",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-7-487a21b36db8>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpreprocessing\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mStandardScaler\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mdata\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'normalizedAmount'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mStandardScaler\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit_transform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Amount'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[0mdata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Amount'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   2686\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2687\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2688\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2689\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2690\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_getitem_column\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m_getitem_column\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   2693\u001b[0m         \u001b[1;31m# get column\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2694\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mis_unique\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2695\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_item_cache\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2696\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2697\u001b[0m         \u001b[1;31m# duplicate columns & possible reduce dimensionality\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m_get_item_cache\u001b[1;34m(self, item)\u001b[0m\n\u001b[0;32m   2487\u001b[0m         \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcache\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2488\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mres\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2489\u001b[1;33m             \u001b[0mvalues\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_data\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2490\u001b[0m             \u001b[0mres\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_box_item_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mvalues\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2491\u001b[0m             \u001b[0mcache\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mres\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\internals.py\u001b[0m in \u001b[0;36mget\u001b[1;34m(self, item, fastpath)\u001b[0m\n\u001b[0;32m   4113\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4114\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0misna\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4115\u001b[1;33m                 \u001b[0mloc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mitem\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4116\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4117\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0misna\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mitems\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3078\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3079\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3080\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_maybe_cast_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3081\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3082\u001b[0m         \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_indexer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmethod\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmethod\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtolerance\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtolerance\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'Amount'"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "data['normalizedAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))\n",
    "data=data.drop(['Amount'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
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
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Class</th>\n",
       "      <th>normalizedAmount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>0</td>\n",
       "      <td>0.244964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.342475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>...</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>0</td>\n",
       "      <td>1.160686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>0</td>\n",
       "      <td>0.140534</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.073403</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9        ...              V21       V22       V23  \\\n",
       "0  0.098698  0.363787        ...        -0.018307  0.277838 -0.110474   \n",
       "1  0.085102 -0.255425        ...        -0.225775 -0.638672  0.101288   \n",
       "2  0.247676 -1.514654        ...         0.247998  0.771679  0.909412   \n",
       "3  0.377436 -1.387024        ...        -0.108300  0.005274 -0.190321   \n",
       "4 -0.270533  0.817739        ...        -0.009431  0.798278 -0.137458   \n",
       "\n",
       "        V24       V25       V26       V27       V28  Class  normalizedAmount  \n",
       "0  0.066928  0.128539 -0.189115  0.133558 -0.021053      0          0.244964  \n",
       "1 -0.339846  0.167170  0.125895 -0.008983  0.014724      0         -0.342475  \n",
       "2 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752      0          1.160686  \n",
       "3 -1.175575  0.647376 -0.221929  0.062723  0.061458      0          0.140534  \n",
       "4  0.141267 -0.206010  0.502292  0.219422  0.215153      0         -0.073403  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=data.drop(['Time'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
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
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>V10</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Class</th>\n",
       "      <th>normalizedAmount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>0.090794</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>0</td>\n",
       "      <td>0.244964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>-0.166974</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.342475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>0.207643</td>\n",
       "      <td>...</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>0</td>\n",
       "      <td>1.160686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>-0.054952</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>0</td>\n",
       "      <td>0.140534</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>0.753074</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.073403</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 30 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9       V10        ...              V21       V22  \\\n",
       "0  0.098698  0.363787  0.090794        ...        -0.018307  0.277838   \n",
       "1  0.085102 -0.255425 -0.166974        ...        -0.225775 -0.638672   \n",
       "2  0.247676 -1.514654  0.207643        ...         0.247998  0.771679   \n",
       "3  0.377436 -1.387024 -0.054952        ...        -0.108300  0.005274   \n",
       "4 -0.270533  0.817739  0.753074        ...        -0.009431  0.798278   \n",
       "\n",
       "        V23       V24       V25       V26       V27       V28  Class  \\\n",
       "0 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053      0   \n",
       "1  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724      0   \n",
       "2  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752      0   \n",
       "3 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458      0   \n",
       "4 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153      0   \n",
       "\n",
       "   normalizedAmount  \n",
       "0          0.244964  \n",
       "1         -0.342475  \n",
       "2          1.160686  \n",
       "3          0.140534  \n",
       "4         -0.073403  \n",
       "\n",
       "[5 rows x 30 columns]"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "X=data.iloc[:,data.columns!='Class']\n",
    "Y=data.iloc[:,data.columns=='Class']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
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
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>V10</th>\n",
       "      <th>...</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>normalizedAmount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>0.090794</td>\n",
       "      <td>...</td>\n",
       "      <td>0.251412</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>0.244964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>-0.166974</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.069083</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>-0.342475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>0.207643</td>\n",
       "      <td>...</td>\n",
       "      <td>0.524980</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>1.160686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>-0.054952</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.208038</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>0.140534</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>0.753074</td>\n",
       "      <td>...</td>\n",
       "      <td>0.408542</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>-0.073403</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 29 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9       V10        ...              V20       V21  \\\n",
       "0  0.098698  0.363787  0.090794        ...         0.251412 -0.018307   \n",
       "1  0.085102 -0.255425 -0.166974        ...        -0.069083 -0.225775   \n",
       "2  0.247676 -1.514654  0.207643        ...         0.524980  0.247998   \n",
       "3  0.377436 -1.387024 -0.054952        ...        -0.208038 -0.108300   \n",
       "4 -0.270533  0.817739  0.753074        ...         0.408542 -0.009431   \n",
       "\n",
       "        V22       V23       V24       V25       V26       V27       V28  \\\n",
       "0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   \n",
       "1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   \n",
       "2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   \n",
       "3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   \n",
       "4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   \n",
       "\n",
       "   normalizedAmount  \n",
       "0          0.244964  \n",
       "1         -0.342475  \n",
       "2          1.160686  \n",
       "3          0.140534  \n",
       "4         -0.073403  \n",
       "\n",
       "[5 rows x 29 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
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
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Class\n",
       "0      0\n",
       "1      0\n",
       "2      0\n",
       "3      0\n",
       "4      0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## splitting data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(199364, 29)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(199364, 1)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(85443, 29)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train=np.array(X_train)\n",
    "X_test=np.array(X_test)\n",
    "Y_train=np.array(Y_train)\n",
    "X_test=np.array(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Deep neural network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Dell\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Colocations handled automatically by placer.\n",
      "WARNING:tensorflow:From C:\\Users\\Dell\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n"
     ]
    }
   ],
   "source": [
    "model=Sequential([Dense(units=16,input_dim=29,activation='relu'),\n",
    "                  Dense(units=24,activation='relu'),\n",
    "                  Dropout(0.5),\n",
    "                  Dense(units=20,activation='relu'),\n",
    "                  Dense(units=24,activation='relu'),\n",
    "                  Dense(units=1,activation='sigmoid')])\n",
    "                "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_3 (Dense)              (None, 16)                480       \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 24)                408       \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 24)                0         \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 20)                500       \n",
      "_________________________________________________________________\n",
      "dense_6 (Dense)              (None, 24)                504       \n",
      "_________________________________________________________________\n",
      "dense_7 (Dense)              (None, 1)                 25        \n",
      "=================================================================\n",
      "Total params: 1,917\n",
      "Trainable params: 1,917\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Dell\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.cast instead.\n",
      "Epoch 1/5\n",
      "199364/199364 [==============================] - 20s 102us/step - loss: 0.0099 - acc: 0.9979\n",
      "Epoch 2/5\n",
      "199364/199364 [==============================] - 19s 94us/step - loss: 0.0039 - acc: 0.9993\n",
      "Epoch 3/5\n",
      "199364/199364 [==============================] - 19s 94us/step - loss: 0.0035 - acc: 0.9993\n",
      "Epoch 4/5\n",
      "199364/199364 [==============================] - 19s 94us/step - loss: 0.0034 - acc: 0.9994\n",
      "Epoch 5/5\n",
      "199364/199364 [==============================] - 19s 94us/step - loss: 0.0032 - acc: 0.9994\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x186367a7f60>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])\n",
    "model.fit(X_train,Y_train,batch_size=15,epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "85443/85443 [==============================] - 2s 21us/step\n"
     ]
    }
   ],
   "source": [
    "score=model.evaluate(X_test,Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.004301449376001416, 0.9993914071369217]\n"
     ]
    }
   ],
   "source": [
    "print(score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import itertools\n",
    "from sklearn import svm, datasets\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "def plot_confusion_matrix(cm,classes,normalize=False,title='confusion_matrix',cmap=plt.cm.Blues):\n",
    "    if normalize:\n",
    "        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]\n",
    "        print(\"Normalized Confusion Matrix\")\n",
    "    else:\n",
    "        print(\"confusion maatrix without normalized\")\n",
    "    print(cm)\n",
    "    \n",
    "    plt.imshow(cm,interpolation='nearest',cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks=np.arange(len(classes))\n",
    "    plt.xticks(tick_marks,classes,rotation=45)\n",
    "    plt.yticks(tick_marks,classes)\n",
    "    \n",
    "    fmt='.2f' if normalize else 'd'\n",
    "    thresh=cm.max()/2.\n",
    "    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):\n",
    "        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i,j] > thresh else \"black\")\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('predicted label')\n",
    "    plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_pred=model.predict(X_test)\n",
    "Y_test=pd.DataFrame(Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "cnf_matrix=confusion_matrix(Y_test,Y_pred.round())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[85278    18]\n",
      " [   34   113]]\n"
     ]
    }
   ],
   "source": [
    "print(cnf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[85278    18]\n",
      " [   34   113]]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAUkAAAEYCAYAAADRWAT6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xm8VVX9//HXG3AAR4g0BRRSciwUCDCHnAI0Syv5hl8TTIoytaxvg2bFt4zSslTS/H4pSbSccuRnKhJqpV8ZFQdM5TqgN0xAEEdU8PP7Y6+Lh+s55+7LOZd777nvp4/9OHuvvfY664B+XGuvvddSRGBmZsV1au0KmJm1ZQ6SZmZlOEiamZXhIGlmVoaDpJlZGQ6SZmZlOEiamZXhIFkDlPmDpJWS5lRQzoGSHq9m3doCSa9K+mBr18PaJ/lh8vZP0oHAVcBuEfFaa9dnY5F0N/DHiPh9a9fFapdbkrVhZ+CZjhQg85DUpbXrYO2fg2QrkNRH0g2Slkl6UdJFkjpJ+oGkxZKWSrpc0jYpf19JIWmspGclLZd0Vjo3Dvg9sF/qVv5Y0omS7mn0nSFp17R/pKRHJb0i6V+Svp3SD5ZUX3DNHpLulvSSpIWSPl1w7jJJF0v6SypntqRdcvz2kPQ1SYvSdWdL2kXSfZJelnStpE1T3u6Sbkl/TivTfu90biJwIHBR+t0XFZR/iqRFwKLC3y5pU0kLJJ2W0jtLulfSjzbwr9I6gojwthE3oDPwIHA+sAWwOXAAcBJQB3wQ2BK4AbgiXdMXCOB3QFdgAPAmsEc6fyJwT8F3rHec0gLYNe0/DxyY9rsDA9P+wUB92t8k1ef7wKbAocArZF16gMuAFcAQoAvwJ+DqHL8/gGnA1sBe6XfMTL97G+BRYGzK+z7gc0A3YCvgz8BNBWXdDXypSPkzgB5A1yK/fW9gJbAHcBYwC+jc2v9eeGu7m1uSG98QYEfgOxHxWkSsjoh7gOOBX0fEUxHxKnAmMLpRl/HHEfFGRDxIFmgHbGAd3gb2lLR1RKyMiPuL5BlGFqzPiYi3IuJO4BbguII8N0TEnIhYQxYk98n5/edGxMsRsRB4BLgj/e5VwG3AvgAR8WJEXB8Rr0fEK8BE4OM5yv95RKyIiDcan4iIR4CfAjcC3wZOiIi1OettHZCD5MbXB1icAkuhHYHFBceLyVpo2xek/btg/3WyILYhPgccCSyW9DdJ+xXJsyPwXES806hOvapQnxcK9t8ocrwlgKRukv433YJ4Gfg7sK2kzk2U/1wT56eStc5vjYhFOetsHZSD5Mb3HLBTkUGFJWQDMA12AtawfgDJ6zWyLioAkj5QeDIi5kbE0cB2wE3AtUXKWAL0kVT478hOwL82oD4b6r+A3YChEbE1cFBKV/os9WhGU49s/JasVTxC0gEV19JqmoPkxjeH7J7gOZK2kLS5pP3JHuH5pqR+krYEfgZcU6TFmceDwF6S9pG0OfDfDSfS4MXxkraJiLeBl4Fi3c3ZZMH2u5I2kXQw8Cng6g2oz4baiqxl+ZKkHsCERudfILuXmZukE4BBZPdtvw5MTX/eZkU5SG5k6f7Xp4BdgWeBeuDzwBTgCrIu5dPAauC0DfyOJ4CfAH8lG+G9p1GWE4BnUhf2q8AXipTxFvBp4AhgOVnra0xEPLYhddpAF5ANVC0nG2C5vdH5C4Fj08j3pKYKk7RTKnNMRLwaEVcC88gG0cyK8sPkZmZluCVpZlaG30iwqkqvSN5W7FxE+N6ftTvubpuZldGmWpLq0jW06VatXQ1rhn332Km1q2DNsHjxMyxfvlxN58yv89Y7R6x5z3P7RcUby6ZHxMhqfn9La1tBctOt2Gy3/2jtalgz3Dv7otaugjXD/kMHV73MWPNG7v9uVy+4uGfVK9DC2lSQNLP2SKDaHQN2kDSzygjo1NSbou2Xg6SZVU5Vvc3ZpjhImlmF3N02MyvPLUkzsxKEW5JmZqXJLUkzs7I8um1mVkptD9zU7i8zs41DZN3tPFtTRUnfTCtzPiLpqjQpdb+0GuciSdcUrKa5WTquS+f7FpRzZkp/XNKIgvSRKa1O0hl5fp6DpJlVTp3ybeWKkHqRzRY/OCL2JltZdDRwLnB+RPQnW+lyXLpkHLAyInYlmzj53FTOnum6vYCRwG/T8sGdgYvJJpLeEzgu5S3LQdLMKqSqBMmkC9A1rQHVjWypk0OB69L5qcAxaf/odEw6f5gkpfSrI+LNiHiabGnkIWmrSytzvkW2FMnRTVXIQdLMKtdJ+TboKWlewTa+oYiI+BdwHtmyJs8Dq4D5wEsFaz3V8+6Knb1IK2Om86vI1mpfl97omlLpZXngxswq07x3t5dHRNGpiCR1J2vZ9QNeAv5M1jVurGES3GI3OaNMerFGYZMT6jpImlmFqja6fTjwdEQsA5B0A/AxsrXWu6TWYm+y5Y4hawn2AepT93wbYEVBeoPCa0qll+TutplVrjqj288CwyR1S/cWDwMeBe4Cjk15xgI3p/1p6Zh0/s7IllqYBoxOo9/9gP5kSznPBfqn0fJNyQZ3pjVVKbckzaxyVWhJRsRsSdcB9wNrgAeAycBfgKsl/TSlXZouuRS4QlIdWQtydCpnoaRryQLsGuCUtJQzkk4FppONnE+JiIVN1ctB0swqk/MZyDwiYgIwoVHyU2Qj043zrgZGlShnIjCxSPqtwK3NqZODpJlVzq8lmpmVUtuvJTpImlnlPAuQmVkJnk/SzKwcd7fNzMpzd9vMrAyPbpuZlSB3t83MynN328ysNDlImpkVl63e4CBpZlacKD6DY41wkDSzColOnTxwY2ZWkrvbZmZlOEiamZVS4/cka/dGgpltFEJI+bYmy5J2k7SgYHtZ0umSekiaIWlR+uye8kvSJEl1kh6SNLCgrLEp/yJJYwvSB0l6OF0zSU1UzEHSzCrWqVOnXFtTIuLxiNgnIvYBBgGvAzcCZwAzI6I/MDMdQ7aaYv+0jQcuAZDUg2yG86Fks5pPaAisKc/4gutGlv1t+f8YzMyKq1ZLspHDgCcjYjHZUrNTU/pU4Ji0fzRweWRmka2suAMwApgRESsiYiUwAxiZzm0dEfelRcMuLyirKN+TNLPKNO+eZE9J8wqOJ0fE5BJ5RwNXpf3tI+J5gIh4XtJ2Kb0X8FzBNfUprVx6fZH0khwkzaxizWglLo+IwTnK2xT4NHBmU1mLpMUGpJfk7raZVaSaAzcFjgDuj4gX0vELqatM+lya0uuBPgXX9QaWNJHeu0h6SQ6SZlaxFgiSx/FuVxtgGtAwQj0WuLkgfUwa5R4GrErd8unAcEnd04DNcGB6OveKpGFpVHtMQVlFubttZpURqFP1HpSU1A34BPCVguRzgGsljQOe5d31tm8FjgTqyEbCvwgQESsknQ3MTfl+EhEr0v7JwGVAV+C2tJXkIGlmFavmGzcR8TrwvkZpL5KNdjfOG8ApJcqZAkwpkj4P2DtvfRwkzaxifi3RzKyEhoGbWuUgaWaVq90Y6SDZHKcdfwgnfuZjRAQL65YwfsIf+c1Zozlw0K6senU1AON/dAUPPfEvRh8xmG+d+AkAXnvjTb7+s2t4+Il/0X/n7bji3JPWldmv1/s4+5K/cNGVd/ORD/XiN2eNZrPNNmHN2nc4/WfXMG/h4tb4qR3KV750Erfdegvv32475i94BIAHFyzgtFO+ypurV9OlSxcu+M1v+eiQIa1c0zZK7m4bsOP7t+Frx32cfT83kdVvvs0fzz2JUSMGAfD9C27ixr8uWC//M0teZPiXLuClV95g+P57cvEPjuOgMeexaPFSho0+B4BOncST0ycy7a4HAZh4+jFMnHwbd9z7KCMO2JOJpx/DiC9fuHF/aAd0wtgT+erXTuVLJ41Zl3bWmd/lrB9OYMTII7j9tls568zvcsfMu1uvkm2cJ901ALp07kzXzTbh7TVr6br5pjy/bFXJvLMefHrd/pyHnqbX9tu+J88hQ3bj6fplPPv8SgAiYOstNgdgmy27li3fqueAAw9i8TPPrJcmiZdffhmAVatWscOOO7ZCzdqR2m1IOkjmtWTZKi64fCZP3HY2b7z5FjPve4yZsx7j80cM5r9P+RRnfvkI7p7zOD+YNI233l6z3rUnHvMxpt/76HvKHDViENfePn/d8XfOu47/d/Ep/Pybn6FTJ3HIib9q8d9lxf3yVxfwqU+O4MzvfZt33nmHu/7+f61dpTatlrvbLdpGljRS0uNp3rYzmr6i7dp2q64cdfCH2eOoCXxw+Fls0XVTRh/5UX70m2kM+MzZHPCFX9J9my34ry8evt51Bw3uz9hj9uMHF67/UP8mXTrzyY9/mBtmPLAubfyoA/nur26g/xE/5LvnXc8lE47fKL/N3mvy/17CL847n7qnn+MX553PyePHtXaV2qy8b9u010DaYkFSUmfgYrJ3MPcEjpO0Z0t9X0s7dOjuPLPkRZavfJU1a97hpjsfZNiAfvx7edYle+vtNVx+8ywG79V33TV799+RS370n4z65mRWrHptvfJGHLAnCx57jqUrXlmXdvxRQ7lpZnZv8/oZDzB4r51b/odZUX+6YirHfOazAHzu2FHMmzunlWvUtjlIbpghQF1EPBURbwFXk8391i499+8VDPlwP7puvgmQ3U98/OkX+EDPrdfl+fQhH+HRJ7N35ft8oDtXn/dlxv3wcuqeXfqe8v5j5OD1utoAzy9bxYGD+gNw8JAPUffsspb6OdaEHXbckX/8/W8A3H3Xney6a/9WrlHbVstBsiXvSRabz21o40ySxpPNEgybbNmC1anM3EcWc+NfH+C+K7/HmrXv8OBj9Vx6/b3cfNHJ9Oy+FRI89Hg9p028GoAzxx9Bj2234IIzPw/AmrXvcMDxvwCg6+abcOjQ3Tn1p1et9x2nnH0lv/zOsXTp0ok331zznvPWMsZ84Tj+8be7Wb58Obv07c0Pf/RjLr7kd3znW99gzZo1bLb55lx0SakpDw2q++52W6Ps1ccWKFgaBYyIiC+l4xOAIRFxWqlrOnXbLjbb7T9apD7WMlbOvai1q2DNsP/QwcyfP6+qEW2zD/SP3sdPypX3qV8fOT/PfJJtSUu2JEvN52ZmNURAO+1J59KS9yTnAv0l9UuzDI8mm/vNzGpKbY9ut1hLMiLWSDqVbPLLzsCUiFjYUt9nZq2nnca/XFr0YfKIuJVsUkwzq1XKXrGtVX7jxswqImo7SNbuW+lmttFI+bZ8ZWlbSddJekzSPyXtJ6mHpBmSFqXP7imvJE1Kb/U9JGlgQTljU/5FksYWpA+S9HC6ZpKauFnqIGlmFavywM2FwO0RsTswAPgncAYwMyL6AzPTMWRv9PVP23jgklSfHsAEsmezhwATGgJryjO+4LqR5SrjIGlmlcnZiswTIyVtDRwEXAoQEW9FxEtkb+tNTdmmAsek/aOByyMzC9g2LTk7ApgRESsiYiUwAxiZzm0dEfel9XEuLyirKAdJM6tI9pxk7pZkT0nzCrbxjYr7ILAM+IOkByT9XtIWwPZpOVjS53Ypf7E3+3o1kV5fJL0kD9yYWYXUnIGb5U28cdMFGAicFhGzJV3Iu13r4l/+XrEB6SW5JWlmFaviPcl6oD4iZqfj68iC5gupq0z6XFqQv9ibfeXSexdJL8lB0swqU8V7khHxb+A5SbulpMOAR8ne1msYoR4LNEzQOg0Yk0a5hwGrUnd8OjBcUvc0YDMcmJ7OvSJpWBrVHlNQVlHubptZRRruSVbRacCf0uvMTwFfJGvQXStpHPAsMCrlvRU4EqgDXk95iYgVks4mez0a4CcRsSLtnwxcBnQFbktbSQ6SZlaxasbIiFgAFLtveViRvAGcUqKcKcCUIunzgL3z1sdB0swq1l4nr8jDQdLMKuN3t83MSqv1+SQdJM2sQu13rsg8HCTNrGI1HCMdJM2scm5JmpmVIA/cmJmV55akmVkZNRwjHSTNrHJuSZqZldKMpRnaIwdJM6uI/JykmVl5nT26bWZWWg03JB0kzawy2YS6tRslSwbJtGpZSRHxcvWrY2btUQ33tssu37AQeCR9Lmx0/EjLV83M2otqrrst6RlJD0taIGleSushaYakRemze0qXpEmS6iQ9JGlgQTljU/5FksYWpA9K5dela8tWrGSQjIg+EbFT+uzT6HinXL/WzDqEaq1xU+CQiNinYGXFM4CZEdEfmMm7KygeAfRP23jgkqw+6gFMAIYCQ4AJDYE15RlfcN3IchXJtRCYpNGSvp/2e0salOc6M6t9AjpLubYKHA1MTftTgWMK0i+PzCxg27Sa4ghgRkSsiIiVwAxgZDq3dUTcl5Z+uLygrKKaDJKSLgIOAU5ISa8D/9Osn2dmtStnV7sZgzsB3CFpvqTxKW37tNIh6XO7lN4LeK7g2vqUVi69vkh6SXlGtz8WEQMlPZAquCKtYmZmBjSrK92z4T5jMjkiJjfKs39ELJG0HTBD0mPlvrpIWmxAekl5guTbkjo1FCTpfcA7Oa4zsw5AQKf8UXJ5wX3GoiJiSfpcKulGsnuKL0jaISKeT13mpSl7PdCn4PLewJKUfnCj9LtTeu8i+UvKc0/yYuB64P2SfgzcA5yb4zoz6yCqNXAjaQtJWzXsA8PJnqaZBjSMUI8Fbk7704AxaZR7GLAqdcenA8MldU8DNsOB6encK5KGpVHtMQVlFdVkSzIiLpc0Hzg8JY2KCD8CZGZA1Sfd3R64Md2/7AJcGRG3S5oLXCtpHPAsMCrlvxU4EqgjGy/5Iqy7LXg2MDfl+0lErEj7JwOXAV2B29JWUt43bjoDb5N1uXONiJtZx9GM7nZZEfEUMKBI+ovAYUXSAzilRFlTgClF0ucBe+etU57R7bOAq4AdyfrvV0o6M+8XmFntU86tPcrTkvwCMCgiXgeQNBGYD/y8JStmZu1Hh3x3u8DiRvm6AE+1THXMrL3JRrdbuxYtp9wEF+eT3YN8HVgoaXo6Hk42wm1mtu5h8lpVriXZMIK9EPhLQfqslquOmbVHHXJJ2Yi4dGNWxMzapw7b3W4gaRdgIrAnsHlDekR8qAXrZWbtSC13t/M883gZ8Aey/2EcAVwLXN2CdTKzdqaWHwHKEyS7RcR0gIh4MiJ+QDYrkJlZ9saNlGtrj/I8AvRmesfxSUlfBf7Fu9MUmZl1+IXAvglsCXyd7N7kNsBJLVkpM2tfOuTodoOImJ12X+HdiXfNzAAQ7bcrnUe5h8lvpMxklBHx2RapkZm1L81fv6ZdKdeSvGij1SLZd4+duHf2Rv9aM6tQLT8CVO5h8pkbsyJm1n7V8vyJeeeTNDMrSnTQlqSZWV5dargpmfunSdqsJStiZu1Ttn5N9ZaUldRZ0gOSbknH/STNlrRI0jUNq7VK2iwd16XzfQvKODOlPy5pREH6yJRWJ+mMPPXJMzP5EEkPA4vS8QBJv8n1a82sQ+ikfFtO3wD+WXB8LnB+RPQHVgLjUvo4YGVE7Aqcn/IhaU9gNLAXMBL4bQq8nckWNjyCbC6K41Le8r8tR4UnAUcBLwJExIP4tUQzK1DF1RJ7A58Efp+OBRwKXJeyTAWOSftHp2PS+cNS/qOBqyPizYh4mmyRsCFpq4uIpyLiLbI5KI5uqk55gmSniFjcKG1tjuvMrANoWHc757vbPSXNK9jGNyruAuC7wDvp+H3ASxGxJh3XA73Sfi/gOYB0flXKvy690TWl0svKM3DznKQhQKTm6mnAEzmuM7MOonP+rvTyiBhc7ISko4ClETFf0sENyUWyRhPnSqUXaxSWfGGmQZ4geTJZl3sn4AXgrynNzAxVb4af/YFPSzqSbO7arclalttK6pJai72BJSl/PdAHqJfUhWxeiRUF6Q0KrymVXlKT3e2IWBoRoyOiZ9pGR8Typq4zs46jGvckI+LMiOgdEX3JBl7ujIjjgbuAY1O2scDNaX9aOiadvzOtwz0NGJ1Gv/sB/YE5wFygfxot3zR9x7Smfluemcl/R5EmaUQ0vpdgZh1UC08C9D3gakk/BR4AGpaWuRS4QlIdWQtyNEBELJR0LfAosAY4JSLWAkg6FZgOdAamRMTCpr48T3f7rwX7mwOfYf2bn2bWgTUM3FRTRNwN3J32nyIbmW6cZzUwqsT1E8mmdmycfitwa3PqkmeqtGsKjyVdAcxozpeYWW2r4bcSN+i1xH7AztWuiJm1U4LONRwl89yTXMm79yQ7kfX9c73OY2a1r0MvKZueXh9Atq4NwDtp9MjMbJ1aDpJlHwFKAfHGiFibNgdIM3uPak5w0dbkeS1xjqSBLV4TM2uXGrrbVZzgok0pt8ZNwxPuBwBflvQk8BrZn0lEhAOnmXXoNW7mAAN5d8YNM7P3ENClvTYTcygXJAUQEU9upLqYWTvVUVuS75f0rVInI+LXLVAfM2t3RKeiE+/UhnJBsjOwJcWnHTIzAxoWAmvtWrScckHy+Yj4yUariZm1T+145DqPJu9JmpmVI6BzDUfJckHysI1WCzNr16o9C1BbUjJIRsSKjVkRM2u/ajhGbtAsQGZm64h8r+61V7X828xsY1D13t2WtLmkOZIelLRQ0o9Tej9JsyUtknRNWn6BtETDNZLq0vm+BWWdmdIflzSiIH1kSquT1OSMZg6SZlYx5dxyeBM4NCIGAPsAIyUNA84Fzo+I/sBKYFzKPw5YGRG7AuenfEjak2w5h72AkcBvJXVOK75eDBwB7Akcl/KW5CBpZhUR2aS7ebamRObVdLhJ2gI4FLgupU/l3delj07HpPOHpSkejwaujog3I+JpoI5sCYghQF1EPBURbwFXp7wlOUiaWcWqsVriu2Wps6QFwFKypWKeBF5KE+5AtmRsr7Tfi7TmVjq/CnhfYXqja0qll+SBGzOrULPmiuwpaV7B8eSImFyYIa1suI+kbYEbgT2KlNMwt22xL44y6cUahmXnyXWQNLOKNHN0e3lEDM6TMSJeknQ3MAzYtmD6xt7AkpStHugD1EvqAmxDtsRMQ3qDwmtKpRfl7raZVayKo9vvTy1IJHUFDgf+CdwFHJuyjQVuTvvT0jHp/J1pBYVpwOg0+t0P6E82/eNcoH8aLd+UbHBnWrk6uSVpZhWr4rPkOwBT0yh0J+DaiLhF0qPA1ZJ+CjwAXJryXwpcIamOrAU5GiAiFkq6FngUWAOckrrxSDoVmE42ic+UiFhYrkIOkmZWEVVxSdmIeAjYt0j6U2Qj043TVwOjSpQ1EZhYJP1W4Na8dXKQNLOKtddFvvJwkDSzitVuiHSQNLMqqOGGpIOkmVUmewSodqOkg6SZVcwtSTOzktQxJ901M8vD3W0zs3KaMXlFe+QgaWYVc5A0MytDNdzd9gQXVbZ69WoO2G8IQwYOYOCAvTj7xxPWO//Nb5xGz223bKXaWYOvfOkkdtpxOwbts/e6tOuv+zMDB+xFt007MX/eu7N5zZ0zh6GD9mHooH0YMnAAN990Y2tUuc2q5qS7bZGDZJVtttlm3D7jTubc/yCz5y3gjum3M3vWLADmz5vHqpdeauUaGsAJY0/k5ltuXy9tr7325uprb+CAAw9aP33vvbl39jxmz1/AzX+5ndO+9hXWrFmDvauak+62NQ6SVSaJLbfMWopvv/02a95+G0msXbuW75/xHSae84tWrqEBHHDgQfTo0WO9tN332IMP7bbbe/J269aNLl2yO1Nvrl5d0+8pbyjl/Kc9cpBsAWvXrmXooH3YacftOPTwTzBk6FAuufgiPnnUp9lhhx1au3q2AebMns3AAXsxeN8PM+ni/1kXNC09AqR8W3vUYkFS0hRJSyU90lLf0VZ17tyZ2fMXUPdMPfPmzuGef/ydG67/M1879bTWrpptoCFDh3L/gwu55765/PLcn7N69erWrlIbkrcd2T6jZEu2JC8jW8qxw9p222056OMH87e77+KpJ+vYa/dd2W3Xvrz++uvstfuurV092wC777EHW2yxBQsf6XD/7y8t5/3I9nqXosWCZET8nWym4A5l2bJlvJQGZ9544w3unPlX9h04iGfq/83jdc/weN0zdOvWjYWP1bVyTS2vZ55+et1AzeLFi3niicfZuW/f1q1UG+LR7RYmabykeZLmLVu+rLWrU7F/P/88Iw8/hI/u+xEO2O+jHHb4Jzjyk0e1drWskTFfOI6DD9yPJx5/nF369uayKZdy8003skvf3syedR+fPfqTfOrIEQD83733MGTQAIYO2ofRx36GC3/zW3r27NnKv6BtUc6tyXKkPpLukvRPSQslfSOl95A0Q9Ki9Nk9pUvSJEl1kh6SNLCgrLEp/yJJYwvSB0l6OF0zSU2MxClbM6dlSOoL3BIRezeRFYBBgwbHvbPnNZ3RzDbI/kMHM3/+vKo26fb48L7xh5vuypV3v127zy+3WqKkHYAdIuJ+SVsB84FjgBOBFRFxjqQzgO4R8T1JRwKnAUcCQ4ELI2KopB7APGAw2ZKx84FBEbFS0hzgG8AssmUcJkXEbaXq1OotSTNr/6o1cBMRz0fE/Wn/FbKVEnsBRwNTU7apZIGTlH55ZGaRLT27AzACmBERKyJiJTADGJnObR0R96VVFS8vKKsoP8dgZhVrxu3GnpIKu4uTI2Jy8TLVl2xRsNnA9hHxPGSBVNJ2KVsv4LmCy+pTWrn0+iLpJbVYkJR0FXAw2R9KPTAhIi4tf5WZtUfN6L8vL9fdXleetCVwPXB6RLxc5rZhsROxAekltViQjIjjWqpsM2s7RHVXS5S0CVmA/FNE3JCSX5C0Q2pF7gAsTen1QJ+Cy3sDS1L6wY3S707pvYvkL8n3JM2sMlV8TjKNNF8K/DMifl1wahrQMEI9Fri5IH1MGuUeBqxK3fLpwHBJ3dNI+HBgejr3iqRh6bvGFJRVlO9JmlnFqjhcvj9wAvCwpAUp7fvAOcC1ksYBzwKj0rlbyUa264DXgS8CRMQKSWcDc1O+n0REw3PbJ5O97NIVuC1tJTlImlnlqhQlI+KeMqUdViR/AKeUKGsKMKVI+jwg12OJ4CBpZhVrv+9l5+EgaWYVaZgFqFY5SJpZ5RwkzcxKc3fbzKyMdjrBTy4OkmZWsRqOkQ6SZlahvPOgtVMOkmZWkWx0u3ajpIOkmVWsdkOkg6SZVUOGvyZMAAAHdElEQVQNR0kHSTOrmB8BMjMro4ZvSTpImlnlajhGOkiaWWWqPeluW+MgaWaVyTmhbnvlIGlmFavhGOkgaWZVUMNR0mvcmFmF8q663XQklTRF0lJJjxSk9ZA0Q9Ki9Nk9pUvSJEl1kh6SNLDgmrEp/yJJYwvSB0l6OF0zSTlupjpImllFGibdzbPlcBkwslHaGcDMiOgPzEzHAEcA/dM2HrgEsqAKTACGAkOACQ2BNeUZX3Bd4+96DwdJM6uccm5NiIi/AysaJR8NTE37U4FjCtIvj8wsYNu03OwIYEZErIiIlcAMYGQ6t3VE3JfWxrm8oKySfE/SzCrWjDduekqaV3A8OSImN3HN9mkpWNK629ul9F7AcwX56lNaufT6IullOUiaWcWa8QjQ8ogYXK2vLZIWG5BelrvbZlaxKvW2S3khdZVJn0tTej3QpyBfb2BJE+m9i6SX5SBpZpVJD5Pn2TbQNKBhhHoscHNB+pg0yj0MWJW65dOB4ZK6pwGb4cD0dO4VScPSqPaYgrJKcnfbzCpSzdcSJV0FHEx277KebJT6HOBaSeOAZ4FRKfutwJFAHfA68EWAiFgh6Wxgbsr3k4hoGAw6mWwEvStwW9rKcpA0s4pV61nyiDiuxKnDiuQN4JQS5UwBphRJnwfs3Zw6OUiaWcX87raZWRmedNfMrJzajZEOkmZWuRqOkQ6SZlYZyUvKmpmVV7sx0kHSzCpXwzHSQdLMKlfDvW0HSTOrVL4JddsrB0kzq0j2WmJr16LlOEiaWcUcJM3MynB328ysFK+7bWZWWoUT6rZ5DpJmVrkajpIOkmZWMb+WaGZWRu2GSAdJM6uGGo6SDpJmVrFafgRI2TIRbYOkZcDi1q5HC+gJLG/tSliz1Orf2c4R8f5qFijpdrI/rzyWR8TIan5/S2tTQbJWSZpXxQXZbSPw35k18LrbZmZlOEiamZXhILlxTG7tCliz+e/MAN+TNDMryy1JM7MyHCTNzMpwkDQzK8NBsoVI2k3SfpI2kdS5tetj+fjvyhrzwE0LkPRZ4GfAv9I2D7gsIl5u1YpZSZI+FBFPpP3OEbG2tetkbYNbklUmaRPg88C4iDgMuBnoA3xX0tatWjkrStJRwAJJVwJExFq3KK2Bg2TL2Bron/ZvBG4BNgX+U6rhiffaIUlbAKcCpwNvSfojOFDauxwkqywi3gZ+DXxW0oER8Q5wD7AAOKBVK2fvERGvAScBVwLfBjYvDJStWTdrGxwkW8Y/gDuAEyQdFBFrI+JKYEdgQOtWzRqLiCUR8WpELAe+AnRtCJSSBkravXVraK3J80m2gIhYLelPQABnpv/I3gS2B55v1cpZWRHxoqSvAL+U9BjQGTiklatlrchBsoVExEpJvwMeJWudrAa+EBEvtG7NrCkRsVzSQ8ARwCcior6162Stx48AbQRpACDS/Ulr4yR1B64F/isiHmrt+ljrcpA0K0LS5hGxurXrYa3PQdLMrAyPbpuZleEgaWZWhoOkmVkZDpJmZmU4SHZgkl5NnztKuq6JvKdL6tbM8g+WdEve9EZ5TpR0UTO/7xlJedd/NsvFQbLGbMikDOm1vGObyHY60KwgaVYLHCTbCUl9JT0maaqkhyRd19CySy2oH0m6BxglaRdJt0uaL+kfDe8eS+on6T5JcyWd3ajsR9J+Z0nnSXo4fc9pkr5O9t75XZLuSvmGp7Lul/RnSVum9JGpnvcAn83xu4ZI+j9JD6TP3QpO90m/43FJEwqu+YKkOZIWSPpfz9ZjLclBsn3ZDZgcER8BXga+VnBudUQcEBFXky2HelpEDCKb2ea3Kc+FwCUR8VHg3yW+YzzQD9g3fc+fImISsAQ4JCIOSV3aHwCHR8RAskmFvyVpc+B3wKeAA4EP5PhNjwEHRcS+wI/IJituMAQ4HtiHLPgPlrQH2Xyd+0fEPsDalMesRfjd7fbluYi4N+3/Efg6cF46vgYgteg+Bvy5YOrKzdLn/sDn0v4VwLlFvuNw4H8iYg1ARKwokmcYsCdwb/qOTYH7gN2BpyNiUarLH8mCbjnbAFMl9SebEGSTgnMzIuLFVNYNZFPNrQEGAXPTd3cFljbxHWYbzEGyfWn8elTh8WvpsxPwUmpl5SmjMeXMMyMijlsvUdonx7WNnQ3cFRGfkdQXuLvgXLHfK2BqRJzZzO8x2yDubrcvO0naL+0fRzaZ73rSOjpPSxoFoEzDHJb3AqPTfqku6h3AVyV1Sdf3SOmvAFul/VnA/pJ2TXm6SfoQWde5n6RdCurYlG3I1gECOLHRuU9I6iGpK3BMqv9M4FhJ2zXUT9LOOb7HbIM4SLYv/wTGpmm8egCXlMh3PDBO0oPAQuDolP4N4BRJc8mCUzG/B54FHkrX/2dKnwzcJumuiFhGFtCuSnWZBeyeJoQYD/wlDdwszvGbfgH8XNK9ZHM3FrqH7LbAAuD6iJgXEY+S3Q+9I333DGCHHN9jtkE8wUU7kbqit0TE3q1cFbMOxS1JM7My3JI0MyvDLUkzszIcJM3MynCQNDMrw0HSzKwMB0kzszL+P5KH4TpCr2N9AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_confusion_matrix(cnf_matrix,classes=[0,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[284256     59]\n",
      " [   113    379]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Y_pred=model.predict(X)\n",
    "Y_expected=pd.DataFrame(Y)\n",
    "cnf_matrix=confusion_matrix(Y_expected,Y_pred.round())\n",
    "plot_confusion_matrix(cnf_matrix,classes=[0,1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "random_forest= RandomForestClassifier(n_estimators=100)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
       "            max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=1, min_samples_split=2,\n",
       "            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,\n",
       "            oob_score=False, random_state=None, verbose=0,\n",
       "            warm_start=False)"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "random_forest.fit(X_train,Y_train.values.ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred=random_forest.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9995084442259752"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "random_forest.score(X_test,Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import itertools\n",
    "from sklearn import svm, datasets\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "def plot_confusion_matrix(cm,classes,normalize=False,title='confusion_matrix',cmap=plt.cm.Blues):\n",
    "    if normalize:\n",
    "        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]\n",
    "        print(\"Normalized Confusion Matrix\")\n",
    "    else:\n",
    "        print(\"confusion maatrix without normalized\")\n",
    "    print(cm)\n",
    "    \n",
    "    plt.imshow(cm,interpolation='nearest',cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks=np.arange(len(classes))\n",
    "    plt.xticks(tick_marks,classes,rotation=45)\n",
    "    plt.yticks(tick_marks,classes)\n",
    "    \n",
    "    fmt='.2f' if normalize else 'd'\n",
    "    thresh=cm.max()/2.\n",
    "    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):\n",
    "        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i,j] > thresh else \"black\")\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('predicted label')\n",
    "    plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[85290     6]\n",
      " [   36   111]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Cnf_matrix=confusion_matrix(Y_test,y_pred)\n",
    "plot_confusion_matrix(Cnf_matrix,classes=[0,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[284309      6]\n",
      " [    36    456]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "t_exp=random_forest.predict(X)\n",
    "Cnf_matrix=confusion_matrix(Y,t_exp.round())\n",
    "plot_confusion_matrix(Cnf_matrix,classes=[0,1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## undersampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "492\n"
     ]
    }
   ],
   "source": [
    "fraud_indices=np.array(data[data.Class==1].index)\n",
    "numbers_fraud=len(fraud_indices)\n",
    "print(numbers_fraud)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "normal_indices=np.array(data[data.Class==0].index)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "492\n"
     ]
    }
   ],
   "source": [
    "random_indices=np.random.choice(normal_indices,numbers_fraud,replace=False)\n",
    "random_indices=np.array(random_indices)\n",
    "print(len(random_indices))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "984\n"
     ]
    }
   ],
   "source": [
    "under_sample_indices=np.concatenate([fraud_indices,random_indices])\n",
    "print(len(under_sample_indices))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "under_sample_data=data.iloc[under_sample_indices,:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_sample=under_sample_data.iloc[:,under_sample_data.columns!='Class']\n",
    "Y_sample=under_sample_data.iloc[:,under_sample_data.columns=='Class']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test=train_test_split(X_sample,Y_sample,test_size=0.3)\n",
    "\n",
    "X_train=np.array(X_train)\n",
    "X_test=np.array(X_test)\n",
    "Y_train=np.array(Y_train)\n",
    "X_test=np.array(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_3 (Dense)              (None, 16)                480       \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 24)                408       \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 24)                0         \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 20)                500       \n",
      "_________________________________________________________________\n",
      "dense_6 (Dense)              (None, 24)                504       \n",
      "_________________________________________________________________\n",
      "dense_7 (Dense)              (None, 1)                 25        \n",
      "=================================================================\n",
      "Total params: 1,917\n",
      "Trainable params: 1,917\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "688/688 [==============================] - 0s 705us/step - loss: 0.5857 - acc: 0.9041\n",
      "Epoch 2/5\n",
      "688/688 [==============================] - 0s 68us/step - loss: 0.2294 - acc: 0.9259\n",
      "Epoch 3/5\n",
      "688/688 [==============================] - 0s 88us/step - loss: 0.1915 - acc: 0.9273\n",
      "Epoch 4/5\n",
      "688/688 [==============================] - 0s 96us/step - loss: 0.1578 - acc: 0.9346\n",
      "Epoch 5/5\n",
      "688/688 [==============================] - 0s 72us/step - loss: 0.1530 - acc: 0.9404\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x1864b560518>"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])\n",
    "model.fit(X_train,Y_train,batch_size=15,epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[143   0]\n",
      " [ 23 130]]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAT0AAAEYCAYAAAAu+iEYAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAH29JREFUeJzt3Xm8XfO9//HX+yQiiAhiykAGMYRbSsTcaqMVxFClpUoUV7WGauuHtB4oddFJ9dJqqprUFHOpmTSuyyWRGhNTIkSmiogYElPSz++PtU5scXLOOmftffZeZ7+feazHXtP5fj87Rz6+3/Vd67sUEZiZ1YuGagdgZtaenPTMrK446ZlZXXHSM7O64qRnZnXFSc/M6oqTnpnVFSe9DkCJv0h6S9KkHOXsLunFcsZWCyS9J2lAteOw2iDfnFx8knYHrgM2j4jF1Y6nvUh6ELg6Iq6odixWHG7pdQybAK/WU8LLQlLnasdgtcdJrwok9ZV0i6Q3JL0p6VJJDZLOlDRT0nxJf5W0Vnp+P0khaaSk1yQtkPTT9NgxwBXAzmk37meSjpL08Ap1hqRN0/V9JD0n6V1JcySdmu7fQ9Lskp/ZUtKDkhZJmipp/5JjYyRdJunOtJyJkgZm+O4h6fuSpqU/d56kgZIelfSOpBskdUnPXVvSHenf01vpep/02PnA7sCl6fe+tKT8EyRNA6aVfndJXSQ9JemkdH8nSY9IOquNv0oroojw0o4L0Al4GrgYWAPoCuwGHA1MBwYA3YBbgKvSn+kHBPAnYDVgG+BDYMv0+FHAwyV1fGo73RfApun6PGD3dH1tYLt0fQ9gdrq+ShrPT4AuwJeBd0m60ABjgIXAUKAzcA0wLsP3D+B2oDuwVfo9xqffey3gOWBkeu66wNeB1YE1gRuBv5WU9SBwbBPl3w+sA6zWxHffGngL2BL4KfAY0Kna/114ab/FLb32NxToBfy/iFgcER9ExMPA4cBvImJGRLwHjAIOXaGL9rOIeD8iniZJnNu0MYaPgcGSukfEWxHxRBPn7ESSfC+MiI8i4h/AHcBhJefcEhGTImIpSdLbNmP9F0XEOxExFZgC3Jd+77eBu4HPA0TEmxFxc0QsiYh3gfOBL2Yo/4KIWBgR7694ICKmAD8HbgVOBY6IiGUZ47YOwEmv/fUFZqaJolQvYGbJ9kySFtQGJfv+VbK+hCQptcXXgX2AmZL+R9LOTZzTC5gVEf9eIabeZYjn9ZL195vY7gYgaXVJf0y7/O8ADwE9JHVqofxZLRwfS9J6visipmWM2ToIJ732NwvYuImL7HNJBiQabQws5dMJIavFJF1CACRtWHowIh6PiAOA9YG/ATc0UcZcoK+k0v9GNgbmtCGetvoxsDmwY0R0B76Q7lf6ubJbD1q6JeH3JK3WvSTtljtKKxQnvfY3ieSa2oWS1pDUVdKuJLec/FBSf0ndgP8Crm+iRZjF08BWkraV1BU4p/FAejH/cElrRcTHwDtAU927iSTJ8zRJq0jaA9gPGNeGeNpqTZKW3yJJ6wBnr3D8dZJrgZlJOgLYnuS658nA2PTv2+qEk147S68f7QdsCrwGzAa+CVwJXEXShXsF+AA4qY11vAScCzxAMoL58AqnHAG8mnYZjwe+3UQZHwH7A3sDC0haR0dGxAttiamNfksycLOAZMDhnhWOXwIcnI7s/q6lwiRtnJZ5ZES8FxHXApNJBpWsTvjmZDOrK27pmVld8R3rVlbpI3F3N3UsInztzKrO3Vszqys11dJT59VCXdasdhjWCp/fcuNqh2CtMHPmqyxYsEAtn5ldp+6bRCz9zH3gTYr337g3IoaXs/7Wqq2k12VNVt38G9UOw1rhkYmXVjsEa4VddxxS9jJj6fuZ/91+8NRlPcseQCt5IMPMchKoIdvSUknSlemEG1OaOHZqOnlEz3Rbkn4nabqkZyRtlyVaJz0zy0dAQ6dsS8vGAJ/p/krqC3yF5N7WRnsDg9LlOOAPWSpw0jOz/KRsSwsi4iGS2XtWdDFwGp9+xPAA4K+ReIzkueyNWqqjpq7pmVkRKVPXNdVT0uSS7dERMbrZ0pN5HOdExNP6dOLszacnl5id7pvXXHlOemaWX4ZWXGpBRGQeTZG0Osm8h19t6nAT+1q8B89Jz8zyEa1p6bXWQKA/0NjK6wM8IWkoScuub8m5fUhmB2qWr+mZWU4Zr+dlbw0uFxHPRsT6EdEvIvqRJLrtIuJfJDNwH5mO4u4EvB0RzXZtwUnPzMqhTKO3kq4DHgU2lzQ7fQfMytwFzCB5rcGfgO9nCdXdWzPLqVUDGc2KiMNaON6vZD2AE1pbh5OemeUj2tR1rRYnPTPLr3IDGWXnpGdmOZWve9senPTMLL8Gd2/NrF40PntbEE56ZpaTu7dmVm88emtmdcUtPTOrG218xKxanPTMLD8PZJhZ/fBAhpnVG3dvzaxuVHY+vbJz0jOznNy9NbN64+6tmdUVj96aWd2Qu7dmVm/cvTWzeiInPTOrF8ls8U56ZlYvRNOv3a5Rxbn6aGY1SjQ0NGRaWixJulLSfElTSvb9UtILkp6RdKukHiXHRkmaLulFSXtlidZJz8xyk5RpyWAMMHyFffcDW0fE54CXgFFpnYOBQ4Gt0p/5vaQW751x0jOz3MqV9CLiIWDhCvvui4il6eZjQJ90/QBgXER8GBGvkLz0e2hLdTjpmVk+asUCPSVNLlmOa2VtRwN3p+u9gVklx2an+5rlgQwzy0Vk7roCLIiIIW2qR/opsBS4ZnnVnxUtleOkZ2a5ZRmkyEPSSGAEMCwiGhPbbKBvyWl9gLktleXurZnlVsaBjKbKHg6cDuwfEUtKDt0OHCppVUn9gUHApJbKc0vPzPIp4316kq4D9iC59jcbOJtktHZV4P40cT4WEcdHxFRJNwDPkXR7T4iIZS3V4aRnZrmV64mMiDisid1/bub884HzW1OHk56Z5dLKgYyqc9Izs9yc9MysfgjU4KRnZnXELT0zqytOemZWNzyQYWb1pzg5z09klMPlZx/OzPEXMPnGn3zm2ClHDOP9Jy9l3R5rADBij/9g0vWjeGzcGTx8zWnssu2A9g7XmnHfvffwua02Z6stNuWXv7iw2uEUgyr7REa5uaVXBlf9/TEuv/5/uOK8Iz+1v88GPfjyTlvw2rxPZsqZMPFF7njwWQC2HtSLqy86mm0P+nm7xmtNW7ZsGaecfAJ33n0/vfv0YbeddmDEiP3ZcvDgaodW8yr97G05FSfSGvbIEy+z8O0ln9n/i1O/zk8v+RufPB8Ni9//aPn6GqutSrQ4J4S1l8cnTWLgwE3pP2AAXbp04ZBvHsodf7+t2mEVQ/apparOLb0K2feL/8Hc+Yt49qU5nzm2/5c+x7kn7c9666zJQSdfXoXorClz586hT59PJu3o3bsPkyZNrGJExVErXdcsKtrSkzQ8nbt+uqQzKllXLVmt6yqcfsxenPuHO5s8fvuEZ9j2oJ/zjR+N5qzv79vO0dnKRBPN7iL9Y66WrNfzauXvsmJJL52r/jJgb2AwcFg6p32HN6DPemzSe10mXT+KF+78Gb3X78Gj157OBuuu+anzHnniZQb06bl8kMOqq3fvPsye/clEvHPmzKZXr15VjKg4ipT0Ktm9HQpMj4gZAJLGkcxp/1wF66wJU6fPZZNho5Zvv3Dnz9j18F/w5qLFDOjbkxmzFgCw7RZ96LJKZ95ctLhaoVqJITvswPTp03j1lVfo1bs3N14/jjFXXVvtsAqhVhJaFpVMek3NX7/jiielc+Qn8+Sv0q2C4VTO2AuOYvftB9GzRzem33Me511+F2P/9miT535t2LZ8a8SOfLx0GR98+DFHnH5lO0drK9O5c2cuvuRS9tt3L5YtW8bIo45m8FZbVTusQvCzt4lM89dHxGhgNEDD6usXcixz5KgxzR7fYt+zl6//eswD/HrMAxWOyNpq+N77MHzvfaodRrHILb1GbZq/3syKRUCBcl5FR28fBwZJ6i+pC8lLeW+vYH1mVhXFGr2tWEsvIpZKOhG4F+gEXBkRUytVn5lVT43ks0wqenNyRNwF3FXJOsysygQNHsgws3ohnPTMrM4UqXvrCQfMLLdyDWRIulLSfElTSvatI+l+SdPSz7XT/ZL0u/Qx12ckbZclVic9M8tHSUsvy5LBGGD4CvvOAMZHxCBgfLoNySOug9LlOOAPWSpw0jOzXJL79MrT0ouIh4CFK+w+ABibro8FDizZ/9dIPAb0kLRRS3X4mp6Z5aTWDGT0lDS5ZHt0+lRWczaIiHkAETFP0vrp/qYede0NzGuuMCc9M8utFTceL4iIIeWqtol9LT7K6u6tmeVT3mt6TXm9sduafs5P97fpUVcnPTPLpZzX9FbidmBkuj4SuK1k/5HpKO5OwNuN3eDmuHtrZrmV6z49SdcBe5Bc+5sNnA1cCNwg6RjgNeCQ9PS7gH2A6cAS4DtZ6nDSM7PcyjWZQEQctpJDw5o4N4ATWluHk56Z5eNnb82snhRtPj0nPTPLqXbmysvCSc/McitQznPSM7P83NIzs7ohD2SYWb1xS8/M6kqBcp6Tnpnl55aemdWPfJMJtDsnPTPLRb5Pz8zqTSeP3ppZPSlQQ89Jz8zySSYILU7WW2nSk9S9uR+MiHfKH46ZFVGBerfNtvSmksw3X/p1GrcD2LiCcZlZgXSIll5E9F3ZMTOzUgXKednekSHpUEk/Sdf7SNq+smGZWVEI6CRlWmpBi0lP0qXAl4Aj0l1LgMsrGZSZFUjGlwLVShc4y+jtLhGxnaQnASJioaQuFY7LzAqkRvJZJlmS3seSGkhfoitpXeDfFY3KzApDQEOBsl6Wa3qXATcD60n6GfAwcFFFozKzQinXy74l/VDSVElTJF0nqauk/pImSpom6fq8Pc0Wk15E/BU4E/gVsBA4JCLG5anUzDqOxklEsyzNl6PewMnAkIjYGugEHErSyLo4IgYBbwHH5Ik30+htWvnHwEet+BkzqxMNUqYlg87AapI6A6sD84AvAzelx8cCB+aKtaUTJP0UuA7oBfQBrpU0Kk+lZtaxKOMC9JQ0uWQ5rrGMiJhD0qN8jSTZvQ38E1gUEUvT02YDvfPEmmUg49vA9hGxBEDS+WkgF+Sp2Mw6jlbcjrIgIoaspIy1gQOA/sAi4EZg7yZOjbbE2ChL0pu5wnmdgRl5KjWzjiMZvS1LUXsCr0TEGwCSbgF2AXpI6py29voAc/NU0tyEAxeTZNQlwFRJ96bbXyUZwTUzW35zchm8BuwkaXXgfWAYMBmYABwMjANGArflqaS5lt6U9HMqcGfJ/sfyVGhmHU85XgEZERMl3QQ8ASwFngRGk+SfcZJ+nu77c556mptwIFfBZlYfyti9JSLOBs5eYfcMYGh5ashwTU/SQOB8YDDQtSS4zcoVhJkVW608V5tFlnvuxgB/IUnoewM3kPStzcyAVt2yUnVZkt7qEXEvQES8HBFnksy6YmaWPJFRvpuTKy7LLSsfKmm7vizpeGAOsH5lwzKzIqmRfJZJlqT3Q6AbyTNx5wNrAUdXMigzK5ZyjN62lxaTXkRMTFff5ZOJRM3MgORl37XSdc2iuZuTb6WZxz0i4qCKRGRmxZJx2qha0VxL79J2iyI1eFAfbrrLU/UVydojLq52CNYKH05/vSLlFumWleZuTh7fnoGYWXEVab65LAMZZmYrJTpIS8/MLKvOBWrqZU56klaNiA8rGYyZFU/y/ovitPSyzJw8VNKzwLR0extJ/13xyMysMBqUbakFWRqlvwNGAG8CRMTT+DE0MytRrrehtYcs3duGiJi5QvN1WYXiMbOCKdp7b7MkvVmShgIhqRNwEvBSZcMysyLpVJyclynpfY+ki7sx8DrwQLrPzAzV0AwqWWR59nY+yQt3zcyaVKCcl2nm5D/RxDO4EXFcE6ebWR2qlZHZLLJ0bx8oWe8KfA2YVZlwzKxoOtxARkRcX7ot6Srg/opFZGaFU6Cc16bH0PoDm5Q7EDMrKEGnAmW9LNf03uKTa3oNwELgjEoGZWbFUc5XQAJI6gFcAWxNknuOBl4Ergf6Aa8C34iIt9pSfrNPZKTvxtgGWC9d1o6IARFxQ1sqM7OOqcyPoV0C3BMRW5Dkn+dJGlrjI2IQMJ4cDa9mk15EBHBrRCxLl5XOpGxm9UtSpiVDOd2BLwB/BoiIjyJiEXAAMDY9bSxwYFtjzfLs7SRJ27W1AjPr2Bq7txlbej0lTS5ZVrz1bQDwBvAXSU9KukLSGsAGETEPIP1s8xsZm3tHRueIWArsBvynpJeBxel3jIhwIjSz1r4jY0FEDGnmeGdgO+CkiJgo6RLKPIbQ3EDGpLTyNjcjzazjE9C5fCMZs4HZJW9hvIkk6b0uaaOImCdpI2B+WytoLukJICJebmvhZlYfynXHSkT8S9IsSZtHxIvAMOC5dBkJXJh+3tbWOppLeutJ+lEzwf2mrZWaWUciGijrfXonAddI6gLMAL5DMv5wg6RjgNeAQ9paeHNJrxPQDcr7bcysY0leDFS+8iLiKaCp637DylF+c0lvXkScW45KzKwDq6Gp4LNo8ZqemVlzBHQqUNZrLumVpSlpZh1fh5hlJSIWtmcgZlZcBcp5ftm3meUjsj3aVSuc9Mwsn4K97NtJz8xyK07Kc9Izs5xEB5tE1MysJQXKeU56ZpZXtrnyaoWTnpnl4tFbM6s7bumZWV0pTspz0jOznNTRXgFpZtYSd2/NrK4UJ+U56ZlZGRSooeekZ2b5JLesFCfrOemZWW5u6ZlZHVHHmETUzCyLonVvi/T0iJnVIiXd2yxLpuKkTpKelHRHut1f0kRJ0yRdn74ass2c9Mwst3ImPeAHwPMl2xcBF0fEIOAt4Jg8sTrpmVluyvinxXKkPsC+wBXptoAvAzelp4wFDswTq6/pldG8ObM54wf/yYL5r6OGBr7x7e9w5LEncMkvzuUf995JgxpYp+d6XPDbP7L+hhtVO9y6dvkPv8LeOw7gjUVLGHL8VQCcdeTOjNh5IP/+d/DGovc57tf3Mm/hYgB+/b092GuH/iz58GOO+/V9PDV9fjXDrymtnES0p6TJJdujI2J0yfZvgdOANdPtdYFFEbE03Z4N9M4Rrlt65dSpc2dOO+sC7nzoCa6/YwLXjvkT0196nmO+dwq3jZ/IrQ88yh57Duf3F19Q7VDr3lX3P8cBZ976qX0X3/RPhn7vanY64RrunjSDUYfvBMBeO/RjYK8ebH30Xzjxkgf43YlfrkbINa0V3dsFETGkZBn9SRkaAcyPiH+WFt1EdZEnVrf0ymj9DTZk/Q02BGCNbmsycNPNeX3ePDbdbMvl57z//pJi3dTUQT0yZQ4bb9D9U/veXfLR8vXVu65CRPJva8TOA7l2fHKJadIL/2Ktbquy4Tpr8K+0FWhk6rpmsCuwv6R9gK5Ad5KWXw9JndPWXh9gbp5KnPQqZM6smTw/5Wm22W4IAL+98Bxuu/E6unXvztib7qpydLYy54zchcP3HMzbiz9k+OnJZaRe63Zj9hvvLj9nzhvv0Wvdbk56KQENZch5ETEKGAUgaQ/g1Ig4XNKNwMHAOGAkcFueeirWvZV0paT5kqZUqo5atXjxe5x87OGcce5FdFszaU2ccsY5TPjni+x30De55so/VjlCW5lzxv4fg464gnETXuD4/bYFmm6YN7YCDbIPY7Q5M54O/EjSdJJrfH/OE20lr+mNAYZXsPya9PHHH/ODYw9nv4O+yVf3OeAzx/f92je4765c/6OydnDDhBc4cLdNAZiz4D36rLfm8mO91+u2fIDDKPt9egAR8WBEjEjXZ0TE0IjYNCIOiYgP84RbsaQXEQ8BCytVfi2KCM788fcZMGhzjvruScv3vzpj+vL1CffeyYBNN6tGeNaCgb16LF/fd6eBvDTrLQDufGwG3xqWXJcdusWGvLP4I3dtSzSO3mZZakHVr+lJOg44DqBX775VjiafJyY9yu03XcdmW27F1/bcGYBTRp3DzdeN5ZWXp9HQ0ECv3htzzkWXVDlSG3vG3uz+ub707N6V6Vcdy3lXP8rwHfozqM/a/DuC115/l5P/+wEA7pn0Cnvt0I+pV36HJR8u5bu/ua/K0dee2khn2VQ96aVD1qMBtt5mu0JfKNl+x114fu57n9n/xWF7VSEaa87IC+/+zL6x905d6fk/vGxCJcMpvgJlvaonPTMrvjLdstIunPTMLLcauVyXSSVvWbkOeBTYXNJsSbkeEjaz2qWMSy2oWEsvIg6rVNlmVjuE34ZmZvWklffgVZuTnpnlVqCc56RnZmVQoKznpGdmOeV6rrbdOemZWS7lmmWlvTjpmVl+TnpmVk/cvTWzuuJbVsysrhQo5znpmVlOtfSMWQZOemaWSzJ6W5ys56RnZrkVJ+U56ZlZORQo6znpmVluvmXFzOpKgS7pVfQVkGZWJ8o1iaikvpImSHpe0lRJP0j3ryPpfknT0s+12xqrk56Z5dI4iWiWJYOlwI8jYktgJ+AESYOBM4DxETEIGJ9ut4mTnpnlU8aXfUfEvIh4Il1/F3ge6A0cAIxNTxsLHNjWcH1Nz8xya8UlvZ6SJpdsj05fA/vZMqV+wOeBicAGETEPksQoaf22xuqkZ2b5Zc96CyJiSIvFSd2Am4FTIuKdcr6Dw91bM8tJmf9kKk1ahSThXRMRt6S7X5e0UXp8I2B+W6N10jOzXBonEc2ytFhW0qT7M/B8RPym5NDtwMh0fSRwW1vjdffWzPIrX+9zV+AI4FlJT6X7fgJcCNyQvj/7NeCQtlbgpGdmuZXriYyIeJiVp9Bh5ajDSc/McivSExlOemaWW4FynpOemeWU8cbjWuGkZ2a5ND6GVhROemaWW3FSnpOemZVBgRp6Tnpmlp8nETWz+lKcnOekZ2b5FSjnOemZWT6SXwFpZvWmODnPSc/M8itQznPSM7P8CtS7ddIzs7yyTxBaC5z0zCyX5DG0akeRnZOemeXmpGdmdcXdWzOrH55ayszqifAtK2ZWbwqU9Zz0zCy3Ij2G5vfemlluyri0WI40XNKLkqZLOqMSsTrpmVl+Zch6kjoBlwF7A4OBwyQNLneoTnpmlpsy/mnBUGB6RMyIiI+AccABZY81IspdZptJegOYWe04KqAnsKDaQVirdNTf2SYRsV45C5R0D8nfVxZdgQ9KtkdHxOi0nIOB4RFxbLp9BLBjRJxYznhraiCj3L+MWiFpckQMqXYclp1/Z9lFxPAyFdVUU7DsrTJ3b82sVswG+pZs9wHmlrsSJz0zqxWPA4Mk9ZfUBTgUuL3cldRU97YDG13tAKzV/DtrZxGxVNKJwL1AJ+DKiJha7npqaiDDzKzS3L01s7ripGdmdcVJz8zqipNehUjaXNLOklZJH6+xAvDvquPzQEYFSDoI+C9gTrpMBsZExDtVDcxWStJmEfFSut4pIpZVOyarDLf0ykzSKsA3gWMiYhhwG8kNl6dJ6l7V4KxJkkYAT0m6FiAilrnF13E56VVGd2BQun4rcAfQBfiWVKCJx+qApDWAE4FTgI8kXQ1OfB2Zk16ZRcTHwG+AgyTtHhH/Bh4GngJ2q2pw9hkRsRg4GrgWOBXoWpr4qhmbVYaTXmX8L3AfcISkL0TEsoi4FugFbFPd0GxFETE3It6LiAXAd4HVGhOfpO0kbVHdCK2c/BhaBUTEB5KuIZkhYlT6j+ZDYANgXlWDs2ZFxJuSvgv8UtILJI9DfanKYVkZOelVSES8JelPwHMkrYcPgG9HxOvVjcxaEhELJD1DMoPvVyJidrVjsvLxLSvtIL0gHun1PatxktYGbgB+HBHPVDseKy8nPbMmSOoaER+0fKYVjZOemdUVj96aWV1x0jOzuuKkZ2Z1xUnPzOqKk14dk/Re+tlL0k0tnHuKpNVbWf4eku7Iun+Fc46SdGkr63tVUtb3r1qdctLrYNrykHz6GNbBLZx2CtCqpGdWi5z0CkJSP0kvSBor6RlJNzW2vNIWzlmSHgYOkTRQ0j2S/inpfxufHU1frfeopMclnbdC2VPS9U6SfiXp2bSekySdTPLc8ARJE9LzvpqW9YSkGyV1S/cPT+N8GDgow/caKun/JD2Zfm5ecrhv+j1elHR2yc98W9IkSU9J+qNnQ7FWiQgvBViAfiTP8u6abl8JnJquvwqcVnLueGBQur4j8I90/XbgyHT9BOC9krKnpOvfA24GOqfb65TU0TNd7wk8BKyRbp8OnAV0BWaRTKslkqca7mjiu+zRuJ9kGq7GuvYEbk7XjyJ5TnldYDVgCjAE2BL4O7BKet7vS77T8hi9eFnZ4mdvi2VWRDySrl8NnAz8Kt2+HiBtce0C3Fgydd+q6eeuwNfT9auAi5qoY0/g8ohYChARC5s4ZydgMPBIWkcX4FFgC+CViJiWxnI1cFwL32ktYKykQSRJfZWSY/dHxJtpWbeQTM21FNgeeDytezVgfgt1mC3npFcsKz4+U7q9OP1sABZFxLYZy1iRMp5zf0Qc9qmd0rYZfnZF5wETIuJrkvoBD5Yca+r7ChgbEaNaWY8Z4Gt6RbOxpJ3T9cNIJif9lEjew/GKpEMAlGicw+8R4NB0/fCV1HEfcLykzunPr5PufxdYM11/DNhV0qbpOatL2gx4AegvaWBJjC1Zi+Q9IpB0aUt9RdI6klYDDkzjHw8cLGn9xvgkbZKhHjPASa9ongdGptMerQP8YSXnHQ4cI+lpYCpwQLr/B8AJkh4nSTZNuQJ4DXgm/flvpftHA3dLmhARb5AkqOvSWB4DtojkAf3jgDvTgYyZGb7TL4ALJD1CMnddqYdJuuFPkVzrmxwRzwFnAveldd8PbJShHjPAEw4URtr1uyMitq5yKGaF5paemdUVt/TMrK64pWdmdcVJz8zqipOemdUVJz0zqytOemZWV/4/nFoDVEfOvCsAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Y_pred=model.predict(X_test)\n",
    "Y_expected=pd.DataFrame(Y_test)\n",
    "cnf_matrix=confusion_matrix(Y_expected,Y_pred.round())\n",
    "plot_confusion_matrix(cnf_matrix,classes=[0,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[284030    285]\n",
      " [    63    429]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Y_pred=model.predict(X)\n",
    "Y_expected=pd.DataFrame(Y)\n",
    "cnf_matrix=confusion_matrix(Y_expected,Y_pred.round())\n",
    "plot_confusion_matrix(cnf_matrix,classes=[0,1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Smote"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_resample,Y_resample= SMOTE().fit_sample(X,Y.values.ravel())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_resample=pd.DataFrame(X_resample)\n",
    "Y_resample=pd.DataFrame(Y_resample)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test=train_test_split(X_resample,Y_resample,test_size=0.3)\n",
    "\n",
    "X_train=np.array(X_train)\n",
    "X_test=np.array(X_test)\n",
    "Y_train=np.array(Y_train)\n",
    "X_test=np.array(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "398041/398041 [==============================] - 44s 110us/step - loss: 0.0327 - acc: 0.9887\n",
      "Epoch 2/5\n",
      "398041/398041 [==============================] - 39s 98us/step - loss: 0.0143 - acc: 0.9962\n",
      "Epoch 3/5\n",
      "398041/398041 [==============================] - 48s 120us/step - loss: 0.0118 - acc: 0.9970\n",
      "Epoch 4/5\n",
      "398041/398041 [==============================] - 31s 78us/step - loss: 0.0104 - acc: 0.9975\n",
      "Epoch 5/5\n",
      "398041/398041 [==============================] - 34s 86us/step - loss: 0.0097 - acc: 0.9978\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x186381754a8>"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])\n",
    "model.fit(X_train,Y_train,batch_size=15,epochs=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[85155   229]\n",
      " [    9 85196]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Y_pred=model.predict(X_test)\n",
    "Y_expected=pd.DataFrame(Y_test)\n",
    "cnf_matrix=confusion_matrix(Y_expected,Y_pred.round())\n",
    "plot_confusion_matrix(cnf_matrix,classes=[0,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "confusion maatrix without normalized\n",
      "[[283544    771]\n",
      " [     2    490]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Y_pred=model.predict(X)\n",
    "Y_expected=pd.DataFrame(Y)\n",
    "cnf_matrix=confusion_matrix(Y_expected,Y_pred.round())\n",
    "plot_confusion_matrix(cnf_matrix,classes=[0,1])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
