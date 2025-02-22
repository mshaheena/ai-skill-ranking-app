
    "# AI-Powered Data Analysis & Prediction Model\n",
    "\n",
    "## 📌 Project Overview\n",
    "This project analyzes AI skill trends using a dataset from Coursera's Global AI Skills Index. \n",
    "It performs data preprocessing, exploratory data analysis (EDA), and machine learning predictions.\n",
    "\n",
    "## 📊 Dataset Used\n",
    "- **Columns:** Country, region, income group, competency ID, percentile rank.\n",
    "- **Target Variable:** Percentile rank (for regression) or percentile category (for classification).\n",
    "- **Goal:** Predict AI skill ranking based on country and income group.\n",
    "\n",
    "## 📌 Models Used\n",
    "- **Random Forest Regressor**: Predicts AI skill ranking.\n",
    "- **SVM & Logistic Regression**: Tested but gave lower accuracy.\n",
    "- **K-Means Clustering**: Groups countries based on AI skills."
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "\n",
    "# 📌 Step 1: Import Necessary Libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔹 First 10 rows of the dataset:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
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
       "      <th>country</th>\n",
       "      <th>iso3</th>\n",
       "      <th>region</th>\n",
       "      <th>incomegroup</th>\n",
       "      <th>iso</th>\n",
       "      <th>percentile_rank</th>\n",
       "      <th>percentile_category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>United Arab Emirates</td>\n",
       "      <td>ARE</td>\n",
       "      <td>High income</td>\n",
       "      <td>AE</td>\n",
       "      <td>statistical-programming</td>\n",
       "      <td>0.864407</td>\n",
       "      <td>Cutting-edge (75%-100%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>United Arab Emirates</td>\n",
       "      <td>ARE</td>\n",
       "      <td>High income</td>\n",
       "      <td>AE</td>\n",
       "      <td>statistics</td>\n",
       "      <td>0.237288</td>\n",
       "      <td>Lagging (0-25%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>United Arab Emirates</td>\n",
       "      <td>ARE</td>\n",
       "      <td>High income</td>\n",
       "      <td>AE</td>\n",
       "      <td>machine-learning</td>\n",
       "      <td>0.355932</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>United Arab Emirates</td>\n",
       "      <td>ARE</td>\n",
       "      <td>High income</td>\n",
       "      <td>AE</td>\n",
       "      <td>software-engineering</td>\n",
       "      <td>0.322034</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>United Arab Emirates</td>\n",
       "      <td>ARE</td>\n",
       "      <td>High income</td>\n",
       "      <td>AE</td>\n",
       "      <td>fields-of-mathematics</td>\n",
       "      <td>0.491525</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>United Arab Emirates</td>\n",
       "      <td>ARE</td>\n",
       "      <td>High income</td>\n",
       "      <td>AE</td>\n",
       "      <td>artificial-intelligence</td>\n",
       "      <td>0.186441</td>\n",
       "      <td>Lagging (0-25%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Argentina</td>\n",
       "      <td>ARG</td>\n",
       "      <td>Latin America &amp; Caribbean</td>\n",
       "      <td>AR</td>\n",
       "      <td>fields-of-mathematics</td>\n",
       "      <td>0.355932</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Argentina</td>\n",
       "      <td>ARG</td>\n",
       "      <td>Latin America &amp; Caribbean</td>\n",
       "      <td>AR</td>\n",
       "      <td>statistical-programming</td>\n",
       "      <td>0.576271</td>\n",
       "      <td>Competitive (50%-75%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Argentina</td>\n",
       "      <td>ARG</td>\n",
       "      <td>Latin America &amp; Caribbean</td>\n",
       "      <td>AR</td>\n",
       "      <td>software-engineering</td>\n",
       "      <td>0.559322</td>\n",
       "      <td>Competitive (50%-75%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Argentina</td>\n",
       "      <td>ARG</td>\n",
       "      <td>Latin America &amp; Caribbean</td>\n",
       "      <td>AR</td>\n",
       "      <td>statistics</td>\n",
       "      <td>0.610169</td>\n",
       "      <td>Competitive (50%-75%)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                country iso3                      region          incomegroup  \\\n",
       "\n",
       "0  AE  statistical-programming         0.864407  Cutting-edge (75%-100%)  \n",
       "1  AE               statistics         0.237288          Lagging (0-25%)  \n",
       "2  AE         machine-learning         0.355932       Emerging (25%-50%)  \n",
       "3  AE     software-engineering         0.322034       Emerging (25%-50%)  \n",
       "4  AE    fields-of-mathematics         0.491525       Emerging (25%-50%)  \n",
       "5  AE  artificial-intelligence         0.186441          Lagging (0-25%)  \n",
       "6  AR    fields-of-mathematics         0.355932       Emerging (25%-50%)  \n",
       "7  AR  statistical-programming         0.576271    Competitive (50%-75%)  \n",
       "8  AR     software-engineering         0.559322    Competitive (50%-75%)  \n",
       "9  AR               statistics         0.610169    Competitive (50%-75%)  "
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 📌 Step 2: Load and Explore Dataset\n",
    "df = pd.read_csv('Coursera AI GSI Percentile and Category.csv')\n",
    "print(\"🔹 First 10 rows of the dataset:\")\n",
    "df.head(10)\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔹 First 10 rows of the dataset:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
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
       "      <th>country</th>\n",
       "      <th>iso3</th>\n",
       "      <th>region</th>\n",
       "      <th>incomegroup</th>\n",
       "      <th>iso</th>\n",
       "      <th>percentile_rank</th>\n",
       "      <th>percentile_category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>350</th>\n",
       "      <td>Vietnam</td>\n",
       "      <td>VNM</td>\n",
       "      <td>East Asia &amp; Pacific</td>\n",
       "      <td>VN</td>\n",
       "      <td>fields-of-mathematics</td>\n",
       "      <td>0.406780</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>351</th>\n",
       "      <td>Vietnam</td>\n",
       "      <td>VNM</td>\n",
       "      <td>East Asia &amp; Pacific</td>\n",
       "      <td>VN</td>\n",
       "      <td>artificial-intelligence</td>\n",
       "      <td>0.457627</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>352</th>\n",
       "      <td>Vietnam</td>\n",
       "      <td>VNM</td>\n",
       "      <td>East Asia &amp; Pacific</td>\n",
       "      <td>VN</td>\n",
       "      <td>statistics</td>\n",
       "      <td>0.474576</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>353</th>\n",
       "      <td>Vietnam</td>\n",
       "      <td>VNM</td>\n",
       "      <td>East Asia &amp; Pacific</td>\n",
       "      <td>VN</td>\n",
       "      <td>statistical-programming</td>\n",
       "      <td>0.525424</td>\n",
       "      <td>Competitive (50%-75%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>354</th>\n",
       "      <td>South Africa</td>\n",
       "      <td>ZAF</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>ZA</td>\n",
       "      <td>fields-of-mathematics</td>\n",
       "      <td>0.101695</td>\n",
       "      <td>Lagging (0-25%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>355</th>\n",
       "      <td>South Africa</td>\n",
       "      <td>ZAF</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>ZA</td>\n",
       "      <td>software-engineering</td>\n",
       "      <td>0.372881</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>356</th>\n",
       "      <td>South Africa</td>\n",
       "      <td>ZAF</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>ZA</td>\n",
       "      <td>artificial-intelligence</td>\n",
       "      <td>0.338983</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>357</th>\n",
       "      <td>South Africa</td>\n",
       "      <td>ZAF</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>ZA</td>\n",
       "      <td>statistical-programming</td>\n",
       "      <td>0.372881</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>358</th>\n",
       "      <td>South Africa</td>\n",
       "      <td>ZAF</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>ZA</td>\n",
       "      <td>statistics</td>\n",
       "      <td>0.203390</td>\n",
       "      <td>Lagging (0-25%)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>359</th>\n",
       "      <td>South Africa</td>\n",
       "      <td>ZAF</td>\n",
       "      <td>Sub-Saharan Africa</td>\n",
       "      <td>ZA</td>\n",
       "      <td>machine-learning</td>\n",
       "      <td>0.389831</td>\n",
       "      <td>Emerging (25%-50%)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          country iso3               region          incomegroup iso  \\\n",
       "\n",
       "350    fields-of-mathematics         0.406780     Emerging (25%-50%)  \n",
       "351  artificial-intelligence         0.457627     Emerging (25%-50%)  \n",
       "352               statistics         0.474576     Emerging (25%-50%)  \n",
       "353  statistical-programming         0.525424  Competitive (50%-75%)  \n",
       "354    fields-of-mathematics         0.101695        Lagging (0-25%)  \n",
       "355     software-engineering         0.372881     Emerging (25%-50%)  \n",
       "356  artificial-intelligence         0.338983     Emerging (25%-50%)  \n",
       "357  statistical-programming         0.372881     Emerging (25%-50%)  \n",
       "358               statistics         0.203390        Lagging (0-25%)  \n",
       "359         machine-learning         0.389831     Emerging (25%-50%)  "
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(\"🔹 First 10 rows of the dataset:\")\n",
    "df.tail(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "#🔹dataset has the following columns🔹:\n",
    "\n",
    "## Column Name\t                          Description\n",
    "\n",
    "  **country**\t                         **Name of the country**\n",
    "\n",
    "🔹iso3\t                        *(3-letter country code (e.g., USA, IND)\n",
    "🔹region\t                    *(World region (e.g., Asia, Europe)\n",
    "🔹percentile_rank              \t*(Rank of the country in AI skills (higher = better)\n",
    "🔹percentile_category\t        *(AI skill level category (e.g., Top 10%, Bottom 50%)"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "## 🔹 Data Preprocessing\n",
    "Before training the models, we need to clean and preprocess the dataset.\n",
    "- Handle missing values.\n",
    "- Convert categorical features into numerical format using One-Hot Encoding.\n",
    "- Standardize numerical features for better model performance."
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🔹 Dataset Info:\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 360 entries, 0 to 359\n",
      "Data columns (total 8 columns):\n",
      " #   Column               Non-Null Count  Dtype  \n",
      "---  ------               --------------  -----  \n",
      " 0   country              360 non-null    object \n",
      " 1   iso3                 360 non-null    object \n",
      " 2   region               360 non-null    object \n",
      " 3   incomegroup          360 non-null    object \n",
      " 4   iso                  360 non-null    object \n",
      " 6   percentile_rank      360 non-null    float64\n",
      " 7   percentile_category  360 non-null    object \n",
      "dtypes: float64(1), object(7)\n",
      "memory usage: 22.6+ KB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "# 📌 Step 3: Check Column Names and Missing Values\n",
    "print(\"\\n🔹 Dataset Info:\")\n",
    "print(df.info())  # Check data types and missing values"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🔹 Dataset shape:\n",
      "(360, 8)\n"
     ]
    }
   ],
   "source": [
    "print(\"\\n🔹 Dataset shape:\")\n",
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🔹 Dataset columns:\n",
      "       'percentile_rank', 'percentile_category'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(\"\\n🔹 Dataset columns:\")\n",
    "print(df.columns)"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🔹 Summary Statistics:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
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
       "      <th>percentile_rank</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>360.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.293936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.250000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.750000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       percentile_rank\n",
       "count       360.000000\n",
       "mean          0.500000\n",
       "std           0.293936\n",
       "min           0.000000\n",
       "25%           0.250000\n",
       "50%           0.500000\n",
       "75%           0.750000\n",
       "max           1.000000"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(\"\\n🔹 Summary Statistics:\")\n",
    "df.describe() # Get numerical data statistics"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🔹 Dataset null:\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "country                0\n",
       "iso3                   0\n",
       "region                 0\n",
       "incomegroup            0\n",
       "iso                    0\n",
       "percentile_rank        0\n",
       "percentile_category    0\n",
       "dtype: int64"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(\"\\n🔹 Dataset null:\")\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
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
       "      <th>country</th>\n",
       "      <th>iso3</th>\n",
       "      <th>region</th>\n",
       "      <th>incomegroup</th>\n",
       "      <th>iso</th>\n",
       "      <th>percentile_rank</th>\n",
       "      <th>percentile_category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>355</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>356</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>357</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>358</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>359</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>360 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "0      False  False   False        False  False          False   \n",
       "1      False  False   False        False  False          False   \n",
       "2      False  False   False        False  False          False   \n",
       "3      False  False   False        False  False          False   \n",
       "4      False  False   False        False  False          False   \n",
       "..       ...    ...     ...          ...    ...            ...   \n",
       "355    False  False   False        False  False          False   \n",
       "356    False  False   False        False  False          False   \n",
       "357    False  False   False        False  False          False   \n",
       "358    False  False   False        False  False          False   \n",
       "359    False  False   False        False  False          False   \n",
       "\n",
       "     percentile_rank  percentile_category  \n",
       "0              False                False  \n",
       "1              False                False  \n",
       "2              False                False  \n",
       "3              False                False  \n",
       "4              False                False  \n",
       "..               ...                  ...  \n",
       "355            False                False  \n",
       "356            False                False  \n",
       "357            False                False  \n",
       "358            False                False  \n",
       "359            False                False  \n",
       "\n",
       "[360 rows x 8 columns]"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "country                0.0\n",
       "iso3                   0.0\n",
       "region                 0.0\n",
       "incomegroup            0.0\n",
       "iso                    0.0\n",
       "percentile_rank        0.0\n",
       "percentile_category    0.0\n",
       "dtype: float64"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(df.isnull().sum()/len(df)*100).sort_values(ascending = False)"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 📌 Step 4: Visualize Missing Data (if any)\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.heatmap(df.isnull(), cmap=\"coolwarm\", cbar=False, yticklabels=False)\n",
    "plt.title(\"Missing Data Heatmap\")  #✅ Add Title\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "# 📌 Step 5: Convert Categorical Data to Numeric (if needed)\n",
    "label_enc = LabelEncoder()\n",
    "\n",
    "for col in df.columns:\n",
    "    if df[col].dtype == \"object\":  # If column is categorical\n",
    "        df[col] = label_enc.fit_transform(df[col])"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABSMAAAMYCAYAAAAw2/LoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABmUklEQVR4nO3dd5hW5bk+7GuGmUGk14CioIlRFBsawdiwJGJsaGI3mijRbduanyYx0WyNLbrTTOwtNlBBsccSsKDGiC02wBIbKirSlA7DzPeHH7ODoM5Q1si853kcHjJrPe9a9/velDXXPGs9ZbW1tbUBAAAAAFjByhu7AAAAAACgNAgjAQAAAIBCCCMBAAAAgEIIIwEAAACAQggjAQAAAIBCCCMBAAAAgEIIIwEAAACAQggjAQAAAIBCCCMBAAAAgEIIIwGAJuGCCy7IuuuuW6//dthhh0atdfDgwVl33XXzySefLHH/tGnTcsYZZ2SHHXbIxhtvnL333jv33HNPg89z+eWXZ911102/fv0yb968zx238LO79dZb633s+fPn57rrrst+++2XzTffPBtttFF23HHH/PKXv8wrr7yy2PiTTz456667bsaNG/eFx11Yy8iRI5Mk7777btZdd90cffTRDT5WQy3sS0M+h2nTpuWvf/1r9tlnn/Tr1y+9e/fOd77znZxxxhmZMGHCcq1veRo/fnzuv//+eo299dZbs+666+aaa65ZsUUBACWhorELAABYHrbYYosce+yxi2y77bbb8t577+WQQw5JmzZt6ra3bt266PLqPP300/nd7373uftnzZqVww47LOPGjcuAAQPSrVu3/P3vf89Pf/rTTJkyJQcffHC9z3XnnXdm1VVXzdSpU/PAAw9kl112WR5vIbNmzcqPfvSjPP/889lwww2z++67p0WLFnn77bdz55135o477shvf/vb7Lnnng0+9sI+rrXWWsul1hXp6aefzgknnJCPPvoovXv3zoABA7LKKqtkzJgxGTJkSG6//fb89a9/zSabbNLYpS7i5Zdfzj777JP9998/O++885eO79WrV4499tiv3PsAAFZOwkgAoEno27dv+vbtu8i2J598Mu+9914OPfTQdO/evZEq+z/33HNPTjnllMyZM+dzx1x33XUZM2ZMTj/99BxwwAFJkqOPPjr7779/fv/732eXXXZJx44dv/RcL730Ul577bUcddRRueKKK3LLLbcstzDyqquuyvPPP5+f//znOfzwwxfZ99prr2X//ffPr3/962y99db1qvU/LamPX0VvvvlmBg0alCS55JJLFptt++CDD+a4447LT37yk/ztb39Lly5dGqPMJfr444+/cKbsZ/Xq1Su9evVagRUBAKXEbdoAACvYlClTctxxx+WnP/1p2rdvnx49enzu2BtuuCGdO3fOvvvuW7etVatW+a//+q/Mnj07d999d73OefvttydJdt555/Tr1y+PP/543nvvvWV6Hws99NBDqaioyCGHHLLYvnXWWScHHXRQ5s6dm4cffni5nO+r6Ne//nVmz56dM888c4m3/e+www4ZNGhQPvnkk1x33XWNUCEAwFeTMBIAKEnvv/9+Tj311GyzzTbp3bt3tt9++5x11lmZMmXKIuMWPp9w0qRJOfHEE7P55ptniy22yNFHH53XXnutXud67bXXMmLEiOy99965/fbb87WvfW2J48aPH58PP/wwffr0SbNmzRbZt3C24FNPPfWl56uurs4999yTjh07Zr311suAAQNSU1OT4cOH16ve+hy/uro6b7311hL377vvvrnooouy5ZZbfuFxXn755XzrW9/KFltsUff8x88+M7IhxowZk6OPPjp9+/bNRhttlD333DM33nhjamtrFxs7cuTI7Lffftlkk02y3Xbb5ZJLLsmCBQvqdZ633347Tz31VNZcc83suuuunzvu4IMPzoknnrjYjNR///vf+elPf5ott9wyvXv3zs4775zzzz8/s2bNWmTcDjvskM0333yx444ePTrrrrtuzj777LptC3+ffvzxxznttNOy1VZbZcMNN8zee++9yLMhL7jggroQ+brrrsu6666b0aNH1z2b889//nNOO+20bLLJJunbt2/uvffez31m5Ntvv52TTjop3/72t9O7d+/ssssuueyyyzJ//vxFxs2YMSPnnHNOBgwYkA033DBbbrlljj322Lz44otf/EEDAE2S27QBgJLzxhtv5MADD8zUqVOz9dZbZ5111smYMWNy/fXX58EHH8xNN9202G21RxxxRD766KN8//vfz4cffpi///3vefLJJzN48OCst956X3i+NddcM3fccUfWXXfdLxw3fvz4uvGf1blz5zRv3jxvvvnml76/Rx55JJMnT85BBx2UsrKy7LzzzvnNb36TW2+9Nccee2zKy5ft59FbbbVVXnnllfz4xz/OIYccku9+97vp2bNn3f7u3bt/6W3xb7/9dg4//PDU1NTk6quvXubbgEeNGpVjjz02lZWV2XnnndO+ffs89thjOf300zN27NiceeaZdWOHDRuWX//61+nYsWP22GOP1NbW5vLLL0/Lli3rda5HH300SfLtb3/7Cz/Lzp0754gjjlhk29NPP53DDz888+fPzw477JDVVlstTz/9dC655JI88sgjGTx4cFZdddWl+AQ+9eMf/zjTpk3LLrvsklmzZuWuu+7K8ccfn8GDB9cF6XvttVduu+22bLzxxtlmm22y+uqr171+6NChKS8vzwEHHJA33ngjm2yySf75z38udp4xY8bk0EMPzZw5c/Ld7343q622Wp555pn88Y9/zFNPPZXLL7+87rM5/vjj89hjj2X77bfPTjvtlEmTJuWee+7Jo48+mltvvTVf//rXl/r9AgArH2EkAFByTjvttEydOjW//e1vs/fee9dtv/zyy/OHP/whZ555Zi644IJFXjNt2rTccccd6dChQ5JPZ9Ydc8wxOfvss3P99dd/4fm6deuWbt26fWld06ZNS5JFFtv5T61atcr06dO/9DgLb9Hefffd64633XbbZeTIkXn00Uez3XbbfekxvsgxxxyTp59+Oi+88EL+8Ic/5A9/+EO6dOmSLbbYIttuu2122mmnLwz2Pvzww/z4xz/OrFmzcuWVV2ajjTZapnpmz56dk08+OW3atMnNN9+c1VZbLUly0kkn5f/9v/+XYcOGZaeddsp2222Xjz/+OOedd166du2aoUOHpmvXrkmSgw46aJFb47/I+++/nyQNXmSnuro6v/zlL7NgwYJceeWV+fa3v50kqampyRlnnJEbb7wxf/nLX3LyySc36Lj/qVmzZrn77rvrAs0tt9wyJ510UoYNG5bNN9+8bobtwjDyuOOOS/LpquXJp48UuP32278wYK+trc3JJ5+c+fPn5+abb14kSD7vvPPy17/+NTfddFMOPPDAvPLKK3nssccycODAnHfeeXXj+vfvn+OPPz633HJLfvGLXyz1+wUAVj5u0wYASsqECRPy5JNPZosttlgkiEySQYMGZa211sqIESPqgsGFjjrqqLogMkl22mmnbLHFFnnyySfzwQcfLJfaqqurkyRVVVVL3F9VVZW5c+d+4TE++eSTPPTQQ+nevXs23XTTuu0Lg8lbbrllmets1apVbrjhhvzqV7+qC6ImTpyYu+++Oz//+c+z44475p577lnia6dNm5bDDjsskydPziWXXJLNNttsmet58MEHM2XKlBx++OF1QWSSlJeX56c//WmS1N2iPmrUqMyYMSOHHHJIXRCZJOutt14GDhxYr/MtDITrO5NyoX/9618ZP358dtttt7ogcmGdJ510Utq2bZtbb711ibeV19dBBx20yMzKhcHz591S/1k9evT40pm+zz//fF599dX84Ac/WGxG63HHHZfKysq6z3vhe3nttdcW+TO10047ZeTIkTnppJPqVRcA0HSYGQkAlJSXX345SZYYgpWXl2fTTTfNm2++mVdffTVbbLFF3b5vfetbi43faKON8uSTT+aVV15ZJNhaWs2bN0+Sz13peN68eV96C++9996befPmZbfddltk+/bbb59WrVrloYceyuTJkxu8yvVnVVZW5tBDD82hhx6aiRMnZvTo0Xn88cfz4IMPZurUqTnxxBPTunXrbLPNNou87uc//3kmTpyY1VZbbZlnRC700ksv1f3/szNak09nCy7s+8JnU/bu3XuxcX369MnQoUO/9Hzt2rVL8umq1A3xRb/3WrVqlXXXXTdPPvlkJkyYsMit0w3x2dmarVu3TvL5v6c+qz6rzo8ZMybJp7faL+nzbtmyZV555ZXU1tZmvfXWS58+ffLss89m2223zbe+9a1ss8022X777b9wIScAoOkSRgIAJWXGjBlJPg1/lmThsyJnz569yPYlLTrTuXPnJKnXrdP10bZt20Vq/KwZM2Z8aYi48BbtSy+9NJdeeukSx9x2220ZNGjQ5x7j3XffzW233bbY9kMPPXSJt5B36dIlu+++e3bffffMnj075513Xm688cZcfvnli4WRU6ZMyXbbbZdRo0blggsuWC636C78/P/2t7997piFweHCz3ZJsxoXfv5fZo011kjyf8/4/CJvvfVWevTokbKysnr/3pszZ0696liSz86qLSsrS5J6z7ZcGIh/kU8++STJp8/OXPj8zCWZOXNmWrVqlauuuipXXHFF7rzzzjz22GN57LHH8tvf/jZbbLFFfvvb39YrAAUAmg5hJABQUhaGUBMnTlzi/oWhVfv27RfZPmfOnLRo0WKRbQtDsIUz5ZbVwkVgFj6/7z9NnDgxc+fO/cLnFL7zzjt59tln07Vr1yU+F3LmzJm5++67c8stt3xhGPnee+/lwgsvXGz7XnvtlTFjxuRXv/pV9ttvv/zXf/3XYmNatGiRU045JXfccccSF9v5zW9+k9133z277bZbrr322uy+++5Zf/31P7eW+lg4W/Saa6750hW8F4apSwqQJ0+eXK/zbb311kmSxx9/PLW1tXWB32d9+OGH2WWXXdKtW7eMHDnyS3/vLQz5/vP305JCxGUJK5eHhZ/32WefnR/84Af1Gn/88cfn+OOPz5tvvpl//OMfueuuu/Lkk0/mpz/9aW6++eYVXTIA8BXimZEAQElZ+Dy8Z599don7n3766VRWVi6yOnSSvPDCC4uN/de//pVmzZotc5i20GqrrVa3KnFNTc0i+5588skkWeQ5kJ+1cFbkgQcemDPOOGOx/37/+9+ne/fuefPNN/P0009/7nH69u2bV155ZbH/unfvns6dO2fChAm5//77v/T9LGk26QYbbJDmzZvn1FNPzYIFC+r+vywW9nTh7cP/adq0aTn77LPrPpsNNtggyZL7P3bs2Hqdr1u3btlyyy3zzjvv5K677vrccUOGDElNTU369euX8vLyuucrLunc8+bNywsvvJCOHTvWBeGVlZWZM2fOYoHk22+/Xa86P8/nhaf19UWf9/z583PuuefWLeo0bty4nHvuuXnuueeSfHob+cEHH5wbbrghPXv2zAsvvFDvW8gBgKZBGAkAlJTVV189W2yxRV588cXFZmRdddVVee2117L99tsvdjvyhRdeuMjt0yNHjszjjz+e/v37L7KwzbLaY4898sEHH2Tw4MF122bMmJFLL700q6yySvbcc8/Pfe2dd96ZsrKy7LrrrkvcX1ZWlr322itJlno22je+8Y307ds3Y8eOzRlnnLHYgjo1NTU5//zzM2vWrMUWCPpP2223Xb7zne9kzJgxX7oa+Zf5zne+k1atWuWKK65YLKj73e9+l+uuu65u+3bbbZcOHTrk+uuvX2Tm5muvvVa36Ep9/PKXv0xFRUVOO+20PPjgg4vtv+2223LFFVekdevWOfbYY5N8+kzKNdZYI/fff38ee+yxurE1NTX53//930ybNi177LFHyss/vURfe+21U11dnUceeaRu7LRp0zJkyJB617kkzZo1S/J/CyY11Oabb5411lgjN998c55//vlF9l1++eW5+uqr8+KLLyb5NJy8+uqrc/HFFy8Sqs6YMSMff/xxOnfu/LkLNgEATZPbtAGAknPGGWfkwAMPzKmnnpr77rsv66yzTsaMGZMnn3wyq6++ek499dTFXjN+/PgMHDgw/fv3z4cffpiRI0emS5cu+dWvfrVca/vJT36S++67L2effXaeeuqprLHGGvn73/+ed955J7/+9a8/N/h8+umnM378+Gy22WZf+Ay+vfbaKxdeeGHuu+++nHrqqXULnDTEH/7whxxyyCEZMmRI7r///myzzTb52te+lo8//jj//Oc/89Zbb2WXXXbJAQcc8IXHOeWUU/KPf/wjf/7zn7PzzjunW7duDa4l+XSRlrPOOisnnXRS9txzz+y0007p0qVLnnzyybz44ovZYIMNcvjhhyf59Db9M888M8cff3z22Wef7Lzzzqmtrc19992XLl261Os5kEmy7rrr5qKLLsrxxx+fo446KhtuuGE22WST1NTU5LnnnsuYMWPSqlWrXHDBBXUrfDdr1iznnntuBg0alCOOOCI77LBDVl999Tz11FMZM2ZM1l9//fz3f/933Tn23XffPPjggznhhBOyxx57pLKyMvfdd1969OhR79Wxl2ThYkv33ntvVl111QwcOHCxRxB8kYXv4yc/+UkOPPDA7LjjjlljjTXy0ksv5Yknnsjqq6+eE088McmnizztvPPOuf/++7PXXnulX79+qa6uzsiRIzN16tScffbZS/0+AICVk5mRAEDJWWuttTJ8+PB8//vfzyuvvJLBgwdnwoQJOeyww3Lrrbcu8fbiP//5z+ndu3eGDx+eZ555JgMHDsywYcOW++IbrVq1ypAhQ/L9738/Tz/9dG644Ya0adMmf/zjH3PwwQd/7uvuvPPOJFlsFe3PWn311dO3b9/MmTPnC28x/iKdO3fOHXfckVNOOSVrr712HnnkkVx11VW57777stpqq+UPf/hDzj///LoZfp+nW7duOfroozNr1qz85je/WapaFtpll10yePDg9OvXL48++mgGDx6cmTNn5qijjsq11167yKIxO+20U6655pqsv/76ueeee/Lwww9nv/32y09/+tMGnbN///655557cthhh2XevHm58847M2zYsMycOTOHHHJI/va3vy32DMvNN988N998c7773e/W9Xf27Nk57rjjcuONNy6yWvr2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AAAAgEIIIwEAAACAQggjAQAAAIBCWMAGAAAAgCbF6ulfXcJIAAAAAJoUq6d/dblNGwAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAAChERWMXUFNTk1tuuSUPPPBAZs6cmfXWWy+DBg1K165dlzi+uro6w4YNy6hRozJr1qysvfba+fGPf5yePXsWWzgAAAAA0CCNPjNy+PDhGTFiRI488sicddZZKSsryznnnJPq6uoljr/yyivz4IMP5sgjj8y5556b1q1b55xzzsmsWbMKrhwAAAAAaIhGDSOrq6tz9913Z5999kmfPn3Ss2fPnHDCCZkyZUpGjx692PiJEyfmwQcfzNFHH50+ffpk9dVXz1FHHZXKysq88cYbjfAOAAAAAID6atQw8q233srs2bPTu3fvum0tW7bMWmutlXHjxi02/rnnnkvLli2zySabLDL+oosuWuQYAAAAAMBXT6M+M3Ly5MlJkk6dOi2yvX379pk0adJi499///106dIlTz75ZG677bZMmTIla6+9dn74wx+me/fuhdQMAAAAACydRg0j586d+2kRFYuWUVVVlZkzZy42fvbs2fnwww8zfPjwHHzwwWnZsmVuvfXWnHbaafnjH/+Ytm3bLnUtn62hMGWNc9pGV5ZUVlY2dhXF0+/Sot+lRb9Li36XFv0uLfpdWvS7tOh3adHvQjUkV2vUMLKqqirJp8+OXPjrJJk3b16aN2++2PiKiorMmjUrxx9/fN1MyBNOOCFHHXVURo0alT322GOpa2nfvv1Sv3ZZjJ85vlHO29gqmlWkc+fOjV1G4fS7tOh3adHv0qLfpUW/S4t+lxb9Li36XVr0+6urUcPIhbdnT5kyJV27dq3bPnXq1PTo0WOx8R06dEizZs0WuSW7qqoqXbp0ycSJE5eplqlTp37uCt4rUvWC4s/5VVC9oDofffRRY5dROP0uLfpdWvS7tOh3adHv0qLfpUW/S4t+lxb9LlZFRUW9J/o1ahjZo0ePtGjRImPHjq0LI2fOnJk333wzAwYMWGz8+uuvnwULFuT111/P17/+9SSfzqL88MMPs9VWWy1TLdXV1Zk/f/4yHWOp1BZ/yq+E2jTO593Y9Lu06Hdp0e/Sot+lRb9Li36XFv0uLfpdWvT7K6tRw8jKysoMGDAgQ4YMSZs2bdK5c+cMHjw4HTt2TN++fVNTU5NPPvkkq666aqqqqrLeeutlww03zIUXXpgjjjgirVu3zrBhw9KsWbNsu+22jflWAAAAAIAv0ahhZJLst99+WbBgQS699NLMmzcvvXr1yimnnJKKiopMnDgxxx57bI4++uj0798/SXLSSSdlyJAh+f3vf5958+Zl3XXXzWmnnZY2bdo07hsBAAAAAL5Qo4eR5eXlOfjgg3PwwQcvtq9Lly4ZNmzYIttatGiRQYMGZdCgQUWVCAAAAAAsB+XL+4CNsQgMAAAAAPDV1+Aw8uGHH/7cfePGjcvPfvazZakHAAAAAGiiGnyb9iWXXJKamprssMMOddvmzJmTwYMHZ8SIEencufNyLRAAAAAAaBoaHEbuvffeufzyy5MkO+ywQ5599tlcccUV+fjjj7PHHntkn332We5FAgAAAAArvwaHkfvtt19atGiRyy67LI888kjGjRuX9dZbL7/61a+yxhprrIgaAQAAAIAmYKlW095jjz3SokWLXHXVVdlss83y85//fHnXBQAAAAA0MfUKI2+55ZYlbu/Zs2eeeeaZXHnllWnXrl3d9h/84AfLpTgAAAAAoOmoVxh58803f+H+ESNGLPK1MBIAAAAA+Kx6hZFDhw5d0XUAAAAAAE1ceWMXAAAAAACUhgYvYFNTU5MHH3wwzz77bObOnZuamppF9peVleV//ud/lluBAAAAAEDT0OAw8oYbbshdd92VLl26pEOHDikvX3RyZW1t7XIrDgAAAABoOhocRo4aNSq77rprDjnkkBVRDwAAAADQRDX4mZFz5szJZptttiJqAQAAAACasAaHkeutt15eeeWVFVELAAAAANCENfg27T333DMXXHBBqqur881vfjNVVVWLjVl//fWXS3EAAAAAQNPR4DDyzDPPTJIMHz78c8cMHTp06SsCAAAAAJqkBoeRp5122oqoAwAAAABo4hocRroFGwAAAABYGg0OI5Pk1VdfzdixY1NdXV23raamJnPnzs3LL7+cs88+e7kVCAAAAAA0DQ0OI++7775cffXVS9xXVlaWjTfeeJmLAgAAAACangaHkffff3822WSTHHfccbn99tsza9as/OhHP8qzzz6biy++ONtss82KqBMAAAAAWMmVN/QFEydOzM4775xWrVrlG9/4Rl5++eVUVVWlX79+GThwYO69994VUScAAAAAsJJrcBhZUVGR5s2bJ0m6deuW999/v+7Zkeutt14mTJiwfCsEAAAAAJqEBoeRPXv2zDPPPJMk6dq1a2pra/Pqq68mSSZPnrx8qwMAAAAAmowGPzNy1113zR/+8IfMmDEjRx99dDbffPNceOGF6devXx599NH06tVrRdQJAAAAAKzkGjwzcosttsgvfvGLdO/ePUly5JFHZrXVVsuIESPSvXv3HHbYYcu9SAAAAABg5dfgmZFJ0qdPn/Tp0ydJ0rp165x66qnLtSgAAAAAoOlp0MzIuXPnZu7cuZ+7/7XXXssvf/nLZS4KAAAAAGh66jUzcs6cObn88svz+OOPJ0n69u2bo48+um5V7U8++SSDBw/OI488krKyshVXLQAAAACw0qpXGHnTTTflH//4R7797W+nRYsWeeSRRzJs2LD88Ic/zOOPP54rr7wyM2fOTK9evTwzEgAAAABYonqFkc8880x22WWX/OhHP0qSfPOb38yNN96Ybt265YorrkiHDh0yaNCgfPvb316RtQIAAAAAK7F6PTNy6tSp2WSTTeq+7tOnT6ZNm5arr74622+/ff70pz8JIgEAAACAL1SvmZHz589Pq1at6r5u2bJlkmTbbbfNkUceuWIqAwAAAACalAatpr3QwkVq+vfvvzxrAQAAAACasKUKIxeqrKxcXnUAAAAAAE1cvW7TTpJp06Zl0qRJSZKamprFtv2nTp06LafyAAAAAICmot5h5O9+97vFtp133nlLHDt06NClrwgAAAAAaJLqFUYeddRRK7oOAAAAAKCJq1cYaaEaAAAAAGBZLdMCNgAAAAAA9SWMBAAAAAAKIYwEAAAAAAohjAQAAAAACrHMYeS8efNSW1u7PGoBAAAAAJqweq2m/VkTJkzI0KFD88ILL2T27Nk555xz8uCDD2b11VfPLrvssrxrBAAAAACagAbPjHzrrbfyy1/+Mm+88Ua22WabulmRFRUVueaaa/Lwww8v7xoBAAAAgCagwTMjr7/++qy99to59dRTkyT3339/kuRHP/pR5s6dm3vvvTf9+/dfrkUCAAAAACu/Bs+MfPXVV7PrrrumWbNmKSsrW2TfVlttlQkTJiy34gAAAACApqPBYWRlZWXmzZu3xH3Tp09PVVXVMhcFAAAAADQ9DQ4jN9poowwbNiyTJ0+u21ZWVpY5c+bkrrvuyoYbbrhcCwQAAAAAmoYGPzPy4IMPzqmnnpoTTjghPXv2TJJcd911mTBhQmpra3PCCScs5xIBAAAAgKagwWFkp06d8rvf/S533313XnrppXTt2jVz5szJ1ltvnd122y3t27dfEXUCAAAAACu5BoeRSdK6desccMABy7sWAAAAAKAJq1cYOWrUqAYddLvttluqYgAAAACApqteYeTFF1/coIMKIwEAAACAz6pXGHnhhReu6DoAAAAAgCauXmFk586dV3QdAAAAAEATV+/btH/wgx+kS5cuX3rLdllZWY466qjlUhwAAAAA0HTUK4wcM2ZMvve979X9GgAAAACgoeoVRl500UVL/DUAAAAAQH2VN/QFF198cSZOnLjEfRMmTMi55567zEUBAAAAAE1PvWZGTpo0qe7Xo0aNyhZbbJHy8sVzzGeffTYvvvji8qsOAAAAAGgy6hVGXnnllfnXv/5V9/Xvfve7zx270UYbLXtVAAAAAECTU68w8ogjjsgLL7yQJLnkkkuy995752tf+9oiY8rLy9OyZctssMEGy79KAAAAAGClV68wskOHDunfv3/d13369EmbNm1WVE0AAAAAQBNUrzDyP/Xv3z+1tbV58803M3fu3NTU1Cw2Zv31118uxQEAAAAATUeDw8h///vf+eMf/5jJkyd/7pihQ4cuU1EAAAAAQNPT4DDy2muvTbNmzXLMMcekQ4cOS1xVGwAAAADgsxocRr7xxhs54YQT8q1vfWtF1AMAAAAANFENntbYtm3blJWVrYhaAAAAAIAmrMFh5M4775w77rgjc+bMWRH1AAAAAABNVINv037//ffz7rvv5ogjjsgaa6yRqqqqRfaXlZXlf/7nf5ZbgQAAAABA09DgMPLDDz9Mz549P3d/bW3tstQDAAAAADRRDQ4jTzvttBVRBwAAAADQxDU4jFxoxowZefnllzNlypT069cvM2bMSLdu3SxuAwAAAAAs0VKFkbfeemtuu+22zJs3L0nyjW98IzfddFOmT5+eU089NS1btlyuRQIAAAAAK78Gr6Z93333ZdiwYdltt91y9tln123/3ve+lw8//DBDhw5drgUCAAAAAE1Dg8PIe++9NwMHDsx+++2Xtddeu277Jptskv333z9PP/30ci0QAAAAAGgaGhxGTpo0Keuvv/4S962++ur5+OOPl7koAAAAAKDpaXAY2bFjx7z66qtL3Pf666+nY8eOy1wUAAAAAND0NHgBmx122CE333xzqqqqstlmmyVJ5syZkyeeeCK33XZbdt999+VeJAAAAACw8mtwGLnnnntm4sSJGTJkSIYMGZIk+c1vfpMk2WabbTJw4MDlWiAAAAAA0DQ0OIwsKyvLEUcckd133z0vvfRSpk+fnpYtW2b99dfPGmussSJqBAAAAACagAaHkUkyYcKEjB07Nt/5zneSJO+++24eeOCB7LLLLunSpctyLRAAAAAAaBoavIDNK6+8kpNPPjn33HNP3bZZs2bl8ccfzy9+8YuMHz9+uRYIAAAAADQNDQ4jb7jhhqy//vo577zz6rZ985vfzIUXXph11103119//XItEAAAAABoGhocRr711lvZddddU1lZucj2ysrK7LLLLnnttdeWW3EAAAAAQNPR4DCyqqoqU6ZMWeK+Tz75JM2aNVvmogAAAACApqfBYWSfPn0ybNiwxZ4N+c4772TYsGHZZJNNlldtAAAAAEAT0uDVtA866KCccsop+fnPf54uXbqkbdu2+eSTT/Lhhx+mS5cu+eEPf7gi6gQAAAAAVnINDiPbtGmT3//+93nooYfy8ssvZ8aMGenRo0cGDBiQ7bffPqusssqKqBMAAAAAWMk1OIy88sors+2222bAgAEZMGDAiqgJAAAAAGiCGvzMyEcffTRz585dEbUAAAAAAE1Yg8PIb3zjG/nXv/61ImoBAAAAAJqwBt+mveaaa+a+++7L6NGj071797Rt23aR/WVlZTnqqKOWW4EAAAAAQNPQ4DDyySefTPv27ZMk7777bt59991F9peVlS2fygAAAACAJqXBYeRFF120IuoAAAAAAJq4BoeRC9XU1OSdd97J1KlT881vfjM1NTVp1arV8qwNAAAAAGhCliqMfOSRR3LDDTdk6tSpSZLf/va3ufnmm9OsWbOccMIJqahY6owTAAAAAGiiGrya9uOPP56LLroovXv3zgknnFC3vW/fvnnuuedyyy23LM/6AAAAAIAmosFTGG+77bZ85zvfyaBBg1JTU1O3vX///vn4448zcuTI7L///su1SAAAAABg5dfgmZETJkzIFltsscR966yzTqZMmbLMRQEAAAAATU+Dw8g2bdrk3XffXeK+d999N23btl3mogAAAACApqfBYeRWW22VYcOG5Yknnsj8+fOTJGVlZXnjjTcyfPjw9OvXb7kXCQAAAACs/Br8zMj99tsv48ePz5/+9KeUlZUlSU4//fTMmTMnvXr18rxIAAAAAGCJGhxGVlZW5le/+lVeeOGFvPTSS5k+fXpatmyZ9ddfP5tuumldQAkAAAAA8J8aHEYutNFGG2WjjTZanrUAAAAAAE1YvcPIkSNH5m9/+1smTZqUr33t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xjfy/vvvL/ciAQAAAICVX73CyLlz52bVVVet+/qNN97IvHnzssEGGywyrqamJmVlZcu3QgAAAACgSahXGNmhQ4d88MEHdV8///zzSZKNNtpokXGvvvpqOnbsuBzLAwAAAACainqFkZtvvnnuuOOOfPDBB3n//fczcuTItGvXLr17964bM3HixNxzzz3ZeOONV1ixAAAAAMDKq14L2Hz/+9/Pc889l+OPPz7Jp4vO/PSnP015+adZ5iWXXJInnngiLVq0yF577bXiqgUAAAAAVlr1CiNbtWqV8847L0888USmTZuWTTbZJGuuuWbd/gkTJmSzzTbLAQcckLZt266wYgEAAACAlVe9wsgkqaqqyrbbbrvEfWeeeeZyKwgAAAAAaJrq9cxIAAAAAIBlJYwEAAAAAAohjAQAAAAACiGMBAAAAAAKIYwEAAAAAApRr9W0J02a1KCDdurUaamKAQAAAACarnqFkcccc0yDDjp06NClKgYAAAAAaLrqFUYeddRR9T7gggULlroYAAAAAKDpqlcY2b9//y8d89FHH2XEiBF5+OGHs+OOOy5rXQAAAABAE1OvMPLz1NbW5plnnsmIESPywgsvpKamJqutttryqg0AAAAAaEKWKoycOnVqHnjggTz44IOZPHlyWrVqlZ122inbbbddvvGNbyzvGgEAAACAJqBBYeQLL7yQv//973nmmWeSJBtssEEmT56cE088Meuvv/4KKRAAAAAAaBrqFUbeeeedGTlyZD788MOsttpq2W+//dK/f/9UVlbmsMMOW9E1AgAAAABNQL3CyCFDhmTNNdfMaaedtsgMyFmzZq2wwgAAAACApqW8PoO22WabfPDBBznnnHNy7rnn5p///Geqq6tXdG0AAAAAQBNSr5mRxx57bAYNGpTHHnssDz30UM4///y0bNky3/rWt5IkZWVlK7RIAAAAAGDlV+8FbFZZZZXstNNO2WmnnfLuu+/mwQcfzGOPPZYkueiii7L11lvn29/+dtZcc80VViwAAAAAsPJq0GraC3Xv3j2HHHJIDj744Dz99NN56KGHcscdd+S2227Lmmuumd/97nfLu04AAAAAYCW3VGHkQuXl5dliiy2yxRZb5OOPP87DDz+cUaNGLa/aAAAAAIAmZJnCyP/Utm3b7Lnnntlzzz2X1yEBAAAAgCakXqtpAwAAAAAsK2EkAAAA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AAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAohDASAAAAACiEMBIAAAAAKIQwEgAAAAAoREVjF1BTU5NbbrklDzzwQGbOnJn11lsvgwYNSteuXZc4fvr06bn66qvzr3/9K0nSr1+/HHrooVlllVWKLBsAAAAAaKBGnxk5fPjwjBgxIkceeWTOOuuslJWV5Zxzzkl1dfUSx//xj3/Mhx9+mF//+tc58cQT88ILL+TKK68suGoAAAAAoKEaNYysrq7O3XffnX322Sd9+vRJz549c8IJJ2TKlCkZPXr0YuNfffXVjBkzJkcffXTWXnvt9O7dO0cccUQeffTRTJkypRHeAQAAAABQX40aRr711luZPXt2evfuXbetZcuWWWuttTJu3LjFxo8bNy7t27fP6quvXrdtgw02SJK8/PLLK75gAAAAAGCpNeozIydPnpwk6dSp0yLb27dvn0mTJi1xfMeOHRfZVlFRkdatWy9xfEM0b948FRXFfxyrLlg1m3bdtPDzNrZVm6+aFi1aNHYZhdPv0qLfpUW/S4t+lxb9Li36XVr0u7Tod2nR72I1a9as3mMbNYycO3fup0V8JgSsqqrKzJkzFxs/b968VFZWLra9srIy8+fPX6Za2rRps0yvX1rt27fPs0c+2yjnpnj6XVr0u7Tod2nR79Ki36VFv0uLfpcW/S4t+v3V1ai3aVdVVSXJYovVzJs3L82bN1/i+CWFjvPnz1/ieAAAAADgq6NRw8iFt2d/dvGZqVOnpkOHDouN79ixY6ZOnbrIturq6kyfPn2x27cBAAAAgK+WRg0je/TokRYtWmTs2LF122bOnJk333wzvXr1Wmx8r169Mnny5HzwwQd121566aUkybrrrrviCwYAAAAAllqjPjOysrIyAwYMyJAhQ9KmTZt07tw5gwcPTseOHdO3b9/U1NTkk08+yaqrrpqqqqqss846WXfddXP++edn0KBBmTNnTq644opst912S5xJCQAAAAB8dZTV1tbWNmYBNTU1ueGGG/Lwww9n3rx56dWrVw4//PB06dIlEydOzLHHHpujjz46/fv3T5J8/PHHueqqq/Kvf/0rVVVV2XLLLXPIIYfUPX8SAAAAAPhqavQwEgAAAAAoDY36zEgAAAAAoHQIIwEAAACAQggjAQAAAIBCCCMBAAAAgEIIIwEAAACAQggjAQAAAIBCCCMBAAAAgEIII6GJq62tXeT/AKxcHnzwwbz22muNXQYFmTdvXmOXQCNwvVZa/rPPeg6UImFkCfrsP3g1NTWNVAlF+OCDD5IkZWVlLnZKyJNPPpkJEyY0dhkUwJ/rpu2vf/1rrrrqqrRr166xS6EAd955Z2699dZMmzatsUuhYK7XSktZWdkiv9ZzoNRUNHYBFGvEiBH597//nWbNmqV79+753ve+l/JymXRTNXr06Pz5z3/OySefnI022qjuYuc/L4BoeoYOHZpbb701xxxzTFZbbbXGLocVbMaMGSkrK0t1dbXAqom55ppr8thjj+Xss89O586d/f1dAl599dW8+OKLadmyZbbddtu0bdu2sUuiAK7XSssDDzyQt99+O5988kk6dOiQ/fbbL82bN2/ssljBPvjgg1RUfBq/dOrUqZGrYUVb+AOm2tradOvWrZGr+WoSRpaQm266KSNGjMgWW2yRSZMm5fnnn8/jjz+e//7v/06XLl0auzxWgPnz52fBggW55JJLcvjhh2fzzTd3gdvE/fWvf83DDz+cbt261c2s0e+m65ZbbsnLL7+cDz74IFVVVdlhhx2yww47ZNVVV23s0lhG119/fUaNGpXzzjuv7t9of46broV/T3ft2jVPPfVUhg0blnnz5uU73/lO2rRp09jlsYK5XisdN954Y0aNGpX+/fsnSZ566qk8++yzOeyww9KrV69UVlY2boGsEMOGDcszzzyTqVOnpkOHDjnggAOy8cYbN3ZZrCA333xznn766UybNi01NTX5/ve/nwEDBjR2WV85wsgS8cEHH+SJJ57Icccdl0022SQ1NTV5/fXXc9lll+W3v/1tjj/++PTs2bOxy2Q5W3vttdOpU6f06NEjV111VWpra/Otb33LhW0Tde211+bRRx/N73//+9x777158skns9tuu5n93ETddtttuf/++3P00Udn3rx5+eijjzJkyJCMHz8+AwcONCt2JbZgwYK88sor6dSpU10QWV1dnZtvvjnvvfdekmSdddbJnnvu2ZhlsgL06tUr8+fPT6dOnTJ48ODU1tZmt912yyqrrNLYpbECuV4rDe+8804ef/zx/Nd//Vc22WSTJMmkSZPyu9/9LldffXUOOuigbLrppmnWrFnjFspydccdd2TkyJE59thjM23atDz88MN57LHHhJFN1PDhwzNixIgcc8wxqampyRtvvJGrr746HTp0yBZbbNHY5X2l+A61RCxYsCAzZ85M165dkyTl5eVZZ511csopp2SVVVbJBRdcUDeLyjMkm44OHTqkqqoqG220UdZbb71cddVVee6555Ik//73vz0kvwm55ppr8uCDD+Z//ud/0qVLl3Tr1i2zZs3KggULGrs0VoA5c+bkpZdeyp577plNN900ffv2zW677ZbDDz88o0aNyt13351JkyY1dpkspWbNmuWQQw5JTU1NbrrppiTJueeemzFjxqRdu3aZP39+HnzwwVx88cWNXCnLy8LQaZVVVsk///nP7L777vn+97+fm2++OSNHjsz555+f6667rpGrZEVxvVYaPv7443zyySd1PyysqalJp06d0qdPn0yYMCHXX3993Q+cfD/WNFRXV+eVV17JHnvskY022ijbbrtt1l577SSfPpbj1VdfbeQKWZ5mzJiRsWPH5pBDDsnGG2+cTTfdNNtvv326d+9e93c6/0cY2cQ9+OCDeeONN9KpU6dUVFTkH//4R92+mpqatG/fPieddFJqamry5z//OUnMolqJ/efFanV1dSoqKtKmTZv07Nkz++yzT3r16pXLLrssv/71rzN8+PBUV1c3YrUsLy+//HJeeOGFnHHGGVlrrbWSJJtuumkmTpyYf/7zn41cHSvC3Llz8/rrry8ya6a2tjbdu3dPmzZt8tBDD+Xuu+9uxApZVj169MhWW22VMWPG5KKLLkrbtm3zi1/8IoMGDcrPfvazbL/99nn99dfrvnFl5VdbW5sePXqkbdu2mTRpUvbdd98cdNBBuf766/Piiy+aUdFELViwIM2aNXO91oTNnTs3SdK2bduUl5fn5ZdfTvJ/33N169YtP/zhD9OqVatcdtlli+xj5VVbW5v58+fn3Xffrfs9kCTPPPNMnn/++fzxj3/M6aefngsvvNAPkJuI+fPn5/XXX1/ke/IOHTqkS5cueeedd/yQ4TP8LdeELVyBs2XLlqmsrEzfvn3z3HPP5dlnn03y6T9ytbW16dixYw4//PBMmjQpTz/9dCNXzdL67AqcFRUVqaioSNeuXfPCCy9ktdVWyz777JOKioq8/vrr2XDDDeueK2cFv5Xbeuutl9NOOy09evRIbW1tampq0rp16/Tq1avugtc/fk3LqquumvXWWy/jxo2ru4AtKyvLKquskn79+uXYY4/Nfffdl6eeeqqRK2VpNW/ePP3790+LFi3y6KOPpnPnzmndunVqampSUVGRHXbYIR988EHefvvtxi6V5aSsrCytWrVKRUVF3WyZiRMnplWrVpkxY0ZeffXVfPzxx41cJcvq+eefz2OPPZZHHnmkLoisrKx0vdZE3Xnnnbntttsybdq0dOzYMWuuuWZGjRqVp59+OtXV1XnnnXdy1VVXpW3btjnuuOPy8ccf51//+ldjl81yUFZWlhYtWmTbbbfNsGHDcu655+aoo45KmzZtcvrpp+ess87KSSedlMcffzz3339/Y5fLctCyZcustdZaee+99zJ37ty6HyS1bNkyFRUVdc8C5lOeGdlE/ecKnF/72teSJDvuuGNeeeWV3H///amqqkrv3r3rZtX07NkzNTU1mThxYmOWzTL4vBU4V1111bqZM3fffXfmz5+fDTfcMPfcc0/at2+fLbfc0jOJVlKPPPJIpkyZkoEDB6Zt27apqalJeXl5XSi1xRZb5K9//WsGDBiQNddc04PwV3KPPPJIpk2blj322COVlZVZf/31c//99+f222/PRhttlKqqqvzlL3/JTjvtlK222irPPvtsxo4dm29961uNXTpLqUOHDjnwwAPz0UcfZeutt07yfz9IXDiLzgrqTcfCv8M7d+6cOXPm5Nprr83zzz+fP//5zxk5cmSGDBmSJJ4FvBK7/vrr849//CPt27fPG2+8kXHjxuXII49M8un12rvvvpvE9VpT8uqrr+aFF17Iqquumt133z0//vGPc8UVV+SKK65IZWVlpk6dmu233z7bbLNNZs+enfnz52fy5MmNXTbL4D+v15Jk++23T0VFRd5///2sssoq2Wuvvepu1e/UqVN+9KMf5bbbbsuuu+6atm3b+nO+kvnPfldVVWXnnXdO586dF1mMas6cOXVh5EKvvfZa1llnncYo+StDGNkELWkFztra2qyxxhr58Y9/nIsuuij3339/pk+fni233DJJ0qpVq3zta19L8+bNG7N0lsKXrcC5xRZb5P77788f//jHvPHGG/nNb36TuXPnZsiQIbn55puz6aabpnnz5v7hW4ks/Inaiy++mOeeey5t27bN9ttvn/Ly8tTU1KSsrCxlZWXZeuut88QTT+T222/PoEGDrLC8kvpsv1u2bJkdd9wxe+yxR6qrq/P888/n4YcfTuvWrbPddtvlwAMPrHvdBx980Jilsxz07Nkz5513XqqqqjJp0qS6P8f3339/pk6dWvcDR1Z+CwPGddddN5dddlm6deuWn/3sZ2nVqlUGDhyY8vLy9OnTRxC5knr44Yfz+OOP55e//GW6dOmS0aNHZ/DgwTnggAPqrtceeugh12tNxGevz4cOHZq5c+dmn332yYknnpjx48dn0qRJ6dq1a3r37p0kqayszBprrOGHTCupz16vtWrVKjvssEM6duyYgQMHZsaMGbnoootSUfFpBLPwB1ALH9PQqlUrf75XIp93fd6vX78sWLBgkX+rZ8+evcjCVDfccEPuuOOOXHHFFWnTpk3htX9VCCObmM9bgXPYsGF5991306FDh6yxxhqZNm1aRowYkXHjxqVXr14ZO3Zs3nzzzbqfzrLyWdIKnHvssUc6deqUf/7zn1lzzTVz8skn133juv/++6dt27ZW6FwJ1dbWpry8PJWVlZk7d27uueeezJs3LzvvvPMigWSLFi2y6aab5qGHHspLL72UzTff3DexK6HP9vu+++7L3Llz873vfS977713+vfvnzlz5qRly5Z1M6KTpKqqKp07d27EylleqqqqMm3atPziF79IeXl5OnbsmOnTp+fnP/95Onbs2NjlsZxtuummeeGFF3LIIYdk9dVXr/uGdeEsG1ZO7777bnr16pUePXok+XQmZPPmzXPLLbdk3rx5WWWVVTJq1Kh07949P//5z12vNRFLuj4fOHBg3Wras2bNymOPPZavf/3rGTVqVN5555307NmzUWtm6Xz2eu3ee+/NvHnzMmDAgCSf/pmvrKzMo48+mu7du9eFzh988EE6dOhQ97x/Vg5Luj6fP39+BgwYkGbNmi0yQWTOnDl1f6ffdNNNue+++3L22WeXdBCZCCObnIUrcF522WW56aabsv/+++fcc8/NnDlz0qNHj3z44YeZO3duamtrs+GGG+axxx7L2LFj06JFi5x22mlmWKyEPrsC5+WXX57Zs2fn5ptvTvPmzTN+/Ph07tw5J5xwQlZfffW6n9QuXOiElc/CQPGDDz7IOuusk7Zt22bEiBEpKyvLd7/73ZSXl9c9h2q33XbLSy+9lMsuuyxdunRxgbsSWlK/H3zwwZSXl2fAgAHp0KFDkmTSpEm55ppr6p4vN3r06Jx55pmNWTrLUbt27XLSSSfl9ddfT/v27fPNb35T2NxErbbaajnppJPqbvHyQ6SV28LZMx9++GHdzJja2trccccdKSsry+zZs/PGG2+kbdu26d27dw477LBFQmjXayunz7s+Hz58eFZdddW8+uqradeuXdZYY43ccsstdbPeTz755HTq1KkxS2cpLel6beTIkSkvL6+7Pl977bUzYsSIXHrppenatWtmzZqVZ555JqeddpofOKxk6tPv6urqlJeXZ/78+WnVqlXuvPPO3HXXXTnzzDPrVlUvZcLIJmjhCpz/+te/6lbgPP7449O6detUV1fnrrvuytNPP52+fftmzz33zJw5c1JeXu4vwJXYklbgXGWVVTJ48OC0atUqJ598clZfffUkMf2/Caitrc3HH3+cefPmZd99903Xrl1z00035e9//3uS5Lvf/W6aNWtW9xPW//f//l/OOuustGjRopErZ2l8Xr//84InSaZMmZIFCxZk9OjR6dq1a04//fR07969katneerVq1d69erV2GVQgP981hQrt4XXXQMHDswrr7ySJPnoo4+y8cYbZ8CAAWnTpk1mz56da6+9Nm+++WZatWqVRAjdFHze9fn111+fVq1a5de//nV69uyZTTbZJLNnz067du1KfqbUyuyLrs9ramoyYMCADBw4MM2bN8+///3vvPbaa+nZs2dOP/30rLHGGo1dPg1Un+/HFs50XWuttXL//fdnlVVWyRlnnCGI/P8JI5ughStwvvzyy3n00UczcODARVbg3HHHHXPLLbfkjTfeyGqrreY5ck3AZ1fg7NSpUyZOnJiWLVtmxowZGTt2bLp06bLILZysvMrKytK6detsu+22dY9k2HvvvXPrrbcu9g9gdXV1qqqqcvrpp/vGZiX1Zf2ura3NzjvvnG9+85v5xje+kerq6pSVlQkzAL5Cvv71r+frX/96kqRLly4ZOHBgqqqqUlNTkxYtWuQHP/hBjjnmmLz55pt1t/Cycvu86/OFdzA899xzad++vVnuTcQXXa+NHDkytbW12WWXXbLLLrtkwYIFda9xfb5yqu/3Y0nqHqlz9tlnmyjwH/zOb6IWrsDZrVu3z12Bc+Gtfaz8ampqkmSxFTj/8pe/5IADDsgNN9yQUaNG1Y1j5desWbPsuOOOWW211VJTU5Pu3btn7733zpprrpm///3vGTFiRJLU/UTOhc7K7Yv6PWLEiLqLnvLy8lRVVQkiAb6iFt62XVVVleT/rs/nz5+fNdZYw/V5E/J51+d//vOfc8ABB+TGG290fd7EfNH12gMPPFB3vdasWbM0a9bM9flK7su+H1vY73333TeXXnqpIPIzzIxswqzAWTqswFmaFj57auEtYAv/Abz99tszfPjwNGvWLDvssENjlshy9EX9vvXWW1NRUaHfAF9xC/8OnzRpUiZOnJg111wz5eXleeSRRzJ37ly36TYhrs9Lk+u10vJl/S4vL89OO+2U9u3bN2aZX0nCyCbOCpylxQqcpek/nwPavXv37L777qmqqsoGG2zQiFWxoug3wMpv8uTJOfvss9OqVau0a9cuM2bMyEknnVS3wi5Nh+vz0uR6rbR8Xr833HDDRqzqq62sduG9AjRp48aNswJniZg/f75bNKlbvIbSoN8AK5/XXnst77zzTlq1apW1117bKspNmOtzEtdrpUa/v5gwEgAAAAAohAdUAAAAAACFEEYCAAAAAIUQRgIAAAAAhRBGAgAAAACFEEYCAAAAAIUQRgIAAAAAhRBGAgAAAACFEEYCANAk1dbWNnYJAAB8RkVjFwAAwMrt9ddfzz333JOxY8fmk08+Sfv27dO7d+/stdde+drXvtYoNT399NN54okncuyxxzbK+QEAWDJhJAAAS+2+++7Ltddemw022CAHHXRQ2rdvnw8++CB33nlnRo8enV//+tdZe+21C6/r7rvvLvycAAB8ObdpAwCwVF5++eVcc8012XnnnXPqqadm6623zgYbbJAdd9wxZ555ZlZZZZVccskljV0mAABfIcJIAACWyp133pmWLVvmgAMOWGxfmzZtcuihh6Zv376ZPXt2kuTxxx/PySefnB/+8If5yU9+kssvvzwzZsyoe82wYcOy7777LnasfffdN8OGDUuSTJw4Mfvuu2/++c9/5g9/+EMOOeSQ/PjHP86ll16aOXPmJElOP/30jB07NmPHjs2+++6bMWPGZMyYMdl3330zYsSIHH300fnJT36Sp556Kvvuu2+ef/75Rc736quvZt99983YsWOX22cFAMCn3KYNAECD1dbW5vnnn8/mm2+e5s2bL3FMv3796n49fPjwDB06NN/97nez//77Z+LEiRk6dGhee+21nH322amqqmrQ+S+//PJsv/32+dnPfpZ///vfuemmm9KmTZsceOCBGTRoUC644IIkyeGHH57u3bvnzTffTJLceOONOeKIIzJv3rxssMEG6dChQx555JFsvPHGdcceNWpUvva1r6VXr14N/VgAAPgSwkgAABps+vTpmT9/frp06fKlY2fMmJFbb701O+ywQwYNGlS3fY011shpp52Whx9+ON/97ncbdP4+ffrkkEMOSZJsuOGGeeGFF/LMM8/kwAMPTPfu3dOiRYskyTe/+c1FXvfd7353kZB02223zb333ps5c+ZklVVWyfz58/P4449n1113TVlZWYNqAgDgy7lNGwCABisv//Qysqam5kvHvvbaa5k/f3622WabRbb36tUrnTt3zksvvdTg8382ZOzYsWPmzp37pa9bc801F/l6hx12yLx58zJ69OgkyVNPPZVZs2Zlu+22a3BNAAB8OWEkAAAN1qpVq7Ro0SIfffTR546ZM2dOZsyYUfdcyHbt2i02pl27dpk1a1aDz//ZW8PLyspSW1v7pa9r27btIl937do1vXr1yiOPPJLk01u0e/func6dOze4JgAAvpwwEgCApbLxxhtnzJgxmTdv3hL3P/zwwzn88MMzadKkJMm0adMWGzN16tS0bt06Sepui/7P2ZYLF6VZkbbffvu89NJLmTBhQl544YX0799/hZ8TAKBUCSMBAFgqu+22W2bMmJGbbrppsX0ff/xx7rjjjnTr1i077bRTKisr8+ijjy4y5uWXX86kSZOy3nrrJUndcx4XhpcLxyyNhbeR10e/fv2yyiqr5IorrkhVVVX69u27VOcEAODLWcAGAICl8s1vfjP77bdfbrrpprz33nvZbrvt0qZNm4wfPz533XVX5syZk1/+8pdp3bp19txzz9xyyy2pqKjIt771rbrVtLt37143E7FPnz657rrrctlll2XPPffMlClTcvPNN9eFlA2x6qqr5rXXXstLL72Unj17fuHY5s2bZ6uttsrIkSOz4447NnhlbwAA6k8YCQDAUtt7772z1lpr5b777su1116bGTNmpEOHDtl0002z9957p1OnTkmSfffdN+3atct9992XBx98MK1bt06/fv2y//771z3/cbXVVsuxxx6b4cOH59xzz83qq6+eI488Mn/9618bXNeAAQPyxhtv5JxzzsnRRx+d9u3bf+H4zTbbLCNHjsz222/f8A8BAIB6K6utz5O+AQCgCbvyyiszbty4/OEPf2jsUgAAmjQzIwEAKFn33HNPJkyYkBEjRuSYY45p7HIAAJo8YSQAACVr3Lhxee6557LLLrtk2223bexyAACaPLdpAwAAAACFKG/sAgAAAACA0iCMBAAAAAAKIYwEAAAAAAohjAQAAAAACiGMBAAAAAAKIYwEAAAAAAohjAQAAAAACiGMBAAAAAAKIYwEAAAAAArx/wHXztuoIWXoSQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1600x900 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 📌 Step 6: Visualize Data Distributions\n",
    "top_countries = df.groupby(\"country\")[\"percentile_rank\"].mean().sort_values(ascending=False).head(10)\n",
    "top_countries.plot(kind=\"bar\", color=\"green\", title=\"Top 10 AI-Skilled Countries\")\n",
    "plt.xlabel(\"Country\")\n",
    "plt.ylabel(\"AI Skill Percentile Rank\")\n",
    "plt.xticks(rotation=45) # ✅ Add Title\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 📌 Step 6: Visualize Data Distributions (Boxplots & Histograms)\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.boxplot(data=df)\n",
    "plt.xticks(rotation=90)\n",
    "plt.title(\"Boxplot of Numerical Features\") # ✅ Add Title\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Define a mapping for region numbers back to names\n",
    "region_mapping = {\n",
    "    0: \"Europe\", 1: \"Asia\", 2: \"Africa\", 3: \"North America\",\n",
    "}\n",
    "\n",
    "# Find the original category\n",
    "df[\"region\"] = df[\"region\"].apply(lambda x: region_mapping[int(x.split(\"_\")[1])])\n",
    "\n",
    "# Now re-run the boxplot\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.boxplot(x=\"region\", y=\"percentile_rank\", data=df)\n",
    "plt.xticks(rotation=45)\n",
    "plt.title(\"AI Skill Distribution Across Regions\") \n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1200x800 with 9 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.hist(figsize=(12, 8), bins=20, color=\"skyblue\")\n",
    "plt.suptitle(\"Histograms of Features\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x600 with 2 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 📌 Step 7: Correlation Heatmap\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.title(\"Correlation Heatmap\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(df.isnull(), yticklabels=False, cbar=True)"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(df.corr(), annot=True, cmap=\"coolwarm\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "x = df.iloc[:,[2,3]].values\n",
    "y = df.iloc[:,4].values"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Latin America & Caribbean', 'High income'],\n",
       "       ['Latin America & Caribbean', 'High income'],\n",
       "       ['Latin America & Caribbean', 'High income'],\n",
       "       ['Latin America & Caribbean', 'High income'],\n",
       "       ['Latin America & Caribbean', 'High income'],\n",
       "       ['Latin America & Caribbean', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['Europe & Central Asia', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['East Asia & Pacific', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
       "       ['North America', 'High income'],\n",
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "array(['AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AR', 'AR', 'AR', 'AR', 'AR',\n",
       "       'AR', 'AU', 'AU', 'AU', 'AU', 'AU', 'AU', 'AT', 'AT', 'AT', 'AT',\n",
       "       'AT', 'AT', 'BE', 'BE', 'BE', 'BE', 'BE', 'BE', 'BD', 'BD', 'BD',\n",
       "       'BD', 'BD', 'BD', 'BY', 'BY', 'BY', 'BY', 'BY', 'BY', 'BR', 'BR',\n",
       "       'BR', 'BR', 'BR', 'BR', 'CA', 'CA', 'CA', 'CA', 'CA', 'CA', 'CH',\n",
       "       'CH', 'CH', 'CH', 'CH', 'CH', 'CL', 'CL', 'CL', 'CL', 'CL', 'CL',\n",
       "       'CN', 'CN', 'CN', 'CN', 'CN', 'CN', 'CO', 'CO', 'CO', 'CO', 'CO',\n",
       "       'CO', 'CR', 'CR', 'CR', 'CR', 'CR', 'CR', 'CZ', 'CZ', 'CZ', 'CZ',\n",
       "       'CZ', 'CZ', 'DE', 'DE', 'DE', 'DE', 'DE', 'DE', 'DK', 'DK', 'DK',\n",
       "       'DK', 'DK', 'DK', 'DO', 'DO', 'DO', 'DO', 'DO', 'DO', 'EC', 'EC',\n",
       "       'EC', 'EC', 'EC', 'EC', 'EG', 'EG', 'EG', 'EG', 'EG', 'EG', 'ES',\n",
       "       'ES', 'ES', 'ES', 'ES', 'ES', 'FI', 'FI', 'FI', 'FI', 'FI', 'FI',\n",
       "       'FR', 'FR', 'FR', 'FR', 'FR', 'FR', 'GB', 'GB', 'GB', 'GB', 'GB',\n",
       "       'GB', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GT', 'GT', 'GT', 'GT',\n",
       "       'GT', 'GT', 'HK', 'HK', 'HK', 'HK', 'HK', 'HK', 'HU', 'HU', 'HU',\n",
       "       'HU', 'HU', 'HU', 'ID', 'ID', 'ID', 'ID', 'ID', 'ID', 'IN', 'IN',\n",
       "       'IN', 'IN', 'IN', 'IN', 'IE', 'IE', 'IE', 'IE', 'IE', 'IE', 'IL',\n",
       "       'IL', 'IL', 'IL', 'IL', 'IL', 'IT', 'IT', 'IT', 'IT', 'IT', 'IT',\n",
       "       'JP', 'JP', 'JP', 'JP', 'JP', 'JP', 'KE', 'KE', 'KE', 'KE', 'KE',\n",
       "       'KE', 'KR', 'KR', 'KR', 'KR', 'KR', 'KR', 'MX', 'MX', 'MX', 'MX',\n",
       "       'MX', 'MX', 'MY', 'MY', 'MY', 'MY', 'MY', 'MY', 'NG', 'NG', 'NG',\n",
       "       'NG', 'NG', 'NG', 'NL', 'NL', 'NL', 'NL', 'NL', 'NL', 'NO', 'NO',\n",
       "       'NO', 'NO', 'NO', 'NO', 'NZ', 'NZ', 'NZ', 'NZ', 'NZ', 'NZ', 'PK',\n",
       "       'PK', 'PK', 'PK', 'PK', 'PK', 'PE', 'PE', 'PE', 'PE', 'PE', 'PE',\n",
       "       'PH', 'PH', 'PH', 'PH', 'PH', 'PH', 'PL', 'PL', 'PL', 'PL', 'PL',\n",
       "       'PL', 'PT', 'PT', 'PT', 'PT', 'PT', 'PT', 'RO', 'RO', 'RO', 'RO',\n",
       "       'RO', 'RO', 'RU', 'RU', 'RU', 'RU', 'RU', 'RU', 'SA', 'SA', 'SA',\n",
       "       'SA', 'SA', 'SA', 'SG', 'SG', 'SG', 'SG', 'SG', 'SG', 'SE', 'SE',\n",
       "       'SE', 'SE', 'SE', 'SE', 'TH', 'TH', 'TH', 'TH', 'TH', 'TH', 'TR',\n",
       "       'TR', 'TR', 'TR', 'TR', 'TR', 'TW', 'TW', 'TW', 'TW', 'TW', 'TW',\n",
       "       'UA', 'UA', 'UA', 'UA', 'UA', 'UA', 'US', 'US', 'US', 'US', 'US',\n",
       "       'US', 'VE', 'VE', 'VE', 'VE', 'VE', 'VE', 'VN', 'VN', 'VN', 'VN',\n",
       "       'VN', 'VN', 'ZA', 'ZA', 'ZA', 'ZA', 'ZA', 'ZA'], dtype=object)"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "# 📌 Step 8: Define Features and Scale Data and split the datasets\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "df = df.drop(columns=[\"country\", \"iso3\", \"iso\"])  "
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_enc = LabelEncoder()\n",
    "df[\"region\"] = label_enc.fit_transform(df[\"region\"])  \n",
    "df[\"incomegroup\"] = label_enc.fit_transform(df[\"incomegroup\"])\n",
    "df[\"percentile_category\"] = label_enc.fit_transform(df[\"percentile_category\"])"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0       3            0  statistical-programming         0.864407   \n",
      "1       3            0               statistics         0.237288   \n",
      "2       3            0         machine-learning         0.355932   \n",
      "3       3            0     software-engineering         0.322034   \n",
      "4       3            0    fields-of-mathematics         0.491525   \n",
      "\n",
      "   percentile_category  \n",
      "0                    1  \n",
      "1                    3  \n",
      "2                    2  \n",
      "3                    2  \n",
      "4                    2  \n"
     ]
    }
   ],
   "source": [
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Scaling successful! Model training can proceed.\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "label_enc = LabelEncoder()\n",
    "#Convert categorical columns to numeric\n",
    "df[\"region\"] = label_enc.fit_transform(df[\"region\"])\n",
    "df[\"incomegroup\"] = label_enc.fit_transform(df[\"incomegroup\"])\n",
    "\n",
    "# Features and target\n",
    "X = df[[\"region\", \"incomegroup\"]]  # Independent variables\n",
    "y = df[\"percentile_rank\"]           # Target variable\n",
    "\n",
    "# Train-Test Split\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Apply StandardScaler AFTER encoding categorical data\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(\"Scaling successful! Model training can proceed.\")"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 0.23324106259251476\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "\n",
    "# Define features (X) and target (y)\n",
    "X = df[[\"region\", \"incomegroup\"]]  \n",
    "y = df[\"percentile_rank\"]\n",
    "\n",
    "# Split into train & test sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train a Linear Regression model\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Predict AI skill ranking\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Evaluate performance\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "print(f\"Mean Absolute Error: {mae}\")"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 38.89%\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Define features & target\n",
    "X = df[[\"region\", \"incomegroup\"]]\n",
    "y = df[\"percentile_category\"]  # Classify into skill levels\n",
    "\n",
    "# Train-Test Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train Random Forest Classifier\n",
    "clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Make Predictions\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "# Evaluate Model\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Model Accuracy: {accuracy * 100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 36.11%\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# Train Random Forest Classifier without scaling\n",
    "clf = RandomForestClassifier(n_estimators=200, random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Make Predictions\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "# Evaluate Model\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Model Accuracy: {accuracy * 100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "percentile_category\n",
      "1    90\n",
      "3    90\n",
      "2    90\n",
      "0    90\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(df[\"percentile_category\"].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "smote = SMOTE(random_state=42)\n",
    "X_resampled, y_resampled = smote.fit_resample(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Initialize LabelEncoder\n",
    "label_enc = LabelEncoder()\n",
    "\n",
    "# Convert categorical columns to numeric\n",
    "df[\"region\"] = label_enc.fit_transform(df[\"region\"])\n",
    "df[\"incomegroup\"] = label_enc.fit_transform(df[\"incomegroup\"])\n",
    "df[\"percentile_category\"] = label_enc.fit_transform(df[\"percentile_category\"])"
   ]
  },
  {
   "cell_type": "code",
   "source": [
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline Accuracy: 27.78%\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "model = LogisticRegression()\n",
    "model.fit(X_train, y_train)\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "from sklearn.metrics import accuracy_score\n",
    "print(f\"Baseline Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(df.corr(), annot=True, cmap=\"coolwarm\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Apply StandardScaler only if needed\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN Accuracy: 29.17%\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Initialize KNN\n",
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "\n",
    "# Train the model\n",
    "knn.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Predict on test data\n",
    "y_pred = knn.predict(X_test_scaled)\n",
    "\n",
    "# Check accuracy\n",
    "from sklearn.metrics import accuracy_score\n",
    "print(f\"KNN Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.2916666666666667\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.25      0.79      0.38        14\n",
      "           1       0.33      0.04      0.08        23\n",
      "           2       0.33      0.11      0.17        18\n",
      "           3       0.37      0.41      0.39        17\n",
      "\n",
      "    accuracy                           0.29        72\n",
      "   macro avg       0.32      0.34      0.25        72\n",
      "weighted avg       0.33      0.29      0.23        72\n",
      "\n",
      "Confusion Matrix:\n",
      "[[11  0  0  3]\n",
      " [20  1  0  2]\n",
      " [ 7  2  2  7]\n",
      " [ 6  0  4  7]]\n"
     ]
    }
   ],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "print(\"Confusion Matrix:\")\n",
    "print(confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 36.11%\n"
     ]
    }
   ],
   "source": [
    "# Initialize and train the model\n",
    "rf = RandomForestClassifier(n_estimators=200, random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions\n",
    "y_pred = rf.predict(X_test)\n",
    "\n",
    "# Evaluate model performance\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Random Forest Accuracy: {accuracy * 100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "feature_importances = rf.feature_importances_"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.barh(range(len(feature_importances)), feature_importances)\n",
    "plt.xlabel('Feature Importance')\n",
    "plt.ylabel('Features')\n",
    "plt.title('Random Forest Importance')\n",
    "plt.yticks(range(len(feature_importances)), X.columns)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x800 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.tree import plot_tree\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Plot one of the trees from the Random Forest\n",
    "plt.figure(figsize=(10, 8))\n",
    "plot_tree(rf.estimators_[0], filled=True)\n",
    "plt.title(\"Random Forest - First Tree\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "df.rename(columns={\"Region\": \"region\"}, inplace=True)  # Rename if needed"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "label_enc = LabelEncoder()\n",
    "\n",
    "# Encode only existing categorical columns\n",
    "for col in df.columns:\n",
    "    if df[col].dtype == \"object\":  # If column contains text\n",
    "        df[col] = label_enc.fit_transform(df[col])\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x800 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Features (X) and Target (y)\n",
    "X = df.drop(columns=[\"percentile_category\"])  # Drop target column\n",
    "y = df[\"percentile_category\"]\n",
    "\n",
    "# Train Decision Tree\n",
    "dt = DecisionTreeClassifier(random_state=42)\n",
    "dt.fit(X, y)\n",
    "\n",
    "# Plot the Decision Tree\n",
    "plt.figure(figsize=(10, 8))\n",
    "plot_tree(dt, filled=True, feature_names=X.columns, class_names=str(y.unique()))\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GaussianNB Accuracy: 30.56%\n"
     ]
    }
   ],
   "source": [
    "# Initialize and train GaussianNB model\n",
    "nb = GaussianNB()\n",
    "nb.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions\n",
    "y_pred = nb.predict(X_test)\n",
    "\n",
    "# Evaluate performance\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"GaussianNB Accuracy: {accuracy * 100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class Conditional Probabilities:\n",
      "[[1.17105263 0.36842105]\n",
      " [1.2238806  0.44776119]\n",
      " [1.72222222 0.94444444]\n",
      " [2.49315068 1.23287671]]\n"
     ]
    }
   ],
   "source": [
    "print(\"Class Conditional Probabilities:\")\n",
    "print(nb.theta_)\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from copy import deepcopy\n",
    "import seaborn as sns\n",
    "sns.set()\n",
    "from matplotlib import pyplot as plt\n",
    "plt.rcParams['figure.figsize'] = (16,9)\n",
    "plt.style.use('ggplot')"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
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
       "      <th>region_2</th>\n",
       "      <th>region_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>355</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>356</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>357</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>358</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>359</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>360 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     region_2  region_3\n",
       "0       False      True\n",
       "1       False      True\n",
       "2       False      True\n",
       "3       False      True\n",
       "4       False      True\n",
       "..        ...       ...\n",
       "355     False     False\n",
       "356     False     False\n",
       "357     False     False\n",
       "358     False     False\n",
       "359     False     False\n",
       "\n",
       "[360 rows x 2 columns]"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "newdf=df.iloc[:,4:6]\n",
    "newdf"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "🔹 Unsupervised Learning Model\n",
    "#⃣## K-Means Clustering\n",
    "✅ Type: Unsupervised Learning\n",
    "✅ Purpose: Groups countries based on their AI skill rankings.\n",
    "✅ Why K-Means?\n",
    "\n",
    "*Finds natural patterns in data\n",
    "*Helps segment countries with similar AI adoption trends"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from sklearn.cluster import KMeans\n",
    "import pandas as pd\n",
    "from sklearn.cluster import KMeans\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "# Encode categorical variables\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_enc = LabelEncoder()\n",
    "for col in df.columns:\n",
    "    if df[col].dtype == \"object\":  # Convert text data into numbers\n",
    "        df[col] = label_enc.fit_transform(df[col])\n",
    "\n",
    "# Features (X) - Select relevant columns\n",
    "X = df.drop(columns=[\"percentile_category\"])  # Replace with actual target column\n",
    "\n",
    "# Scale the features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)  # Now X_scaled is defined!\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n",
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1600x900 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sum_of_squared_distances = []\n",
    "k_range = range(1, 15)  # Define range for k values\n",
    "\n",
    "for k in k_range:\n",
    "    km = KMeans(n_clusters=k, random_state=42)\n",
    "    km.fit(X_scaled)  # Now X_scaled is defined correctly\n",
    "    sum_of_squared_distances.append(km.inertia_)  # Inertia measures clustering quality\n",
    "\n",
    "# Plot Elbow Method Graph\n",
    "plt.plot(k_range, sum_of_squared_distances, marker=\"o\")\n",
    "plt.xlabel(\"Number of Clusters (k)\")\n",
    "plt.ylabel(\"Sum of Squared Distances (Inertia)\")\n",
    "plt.title(\"Elbow Method for Optimal k\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\\anaconda 3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning\n",
      "  warnings.warn(\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   percentile_rank  percentile_category  region_0  region_1  region_2  \\\n",
      "0         0.864407                    1     False     False     False   \n",
      "1         0.237288                    3     False     False     False   \n",
      "2         0.355932                    2     False     False     False   \n",
      "3         0.322034                    2     False     False     False   \n",
      "4         0.491525                    2     False     False     False   \n",
      "\n",
      "   region_3  region_4  region_5  region_6  incomegroup_0  incomegroup_1  \\\n",
      "0      True     False     False     False           True          False   \n",
      "1      True     False     False     False           True          False   \n",
      "2      True     False     False     False           True          False   \n",
      "3      True     False     False     False           True          False   \n",
      "4      True     False     False     False           True          False   \n",
      "\n",
      "0          False            False            False            False   \n",
      "1          False            False            False            False   \n",
      "2          False            False            False             True   \n",
      "3          False            False            False            False   \n",
      "4          False            False             True            False   \n",
      "\n",
      "0            False             True            False        2  \n",
      "1            False            False             True        2  \n",
      "2            False            False            False        2  \n",
      "3             True            False            False        2  \n",
      "4            False            False            False        2  \n"
     ]
    }
   ],
   "source": [
    "# Train KMeans with 3 clusters\n",
    "kmeans = KMeans(n_clusters=3, random_state=42)\n",
    "kmeans.fit(X_scaled)  # Now X_scaled is defined correctly\n",
    "\n",
    "# Get cluster labels\n",
    "labels = kmeans.labels_\n",
    "\n",
    "# Add labels to the original DataFrame\n",
    "df[\"Cluster\"] = labels\n",
    "print(df.head())  # Check assigned clusters\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "centers = kmeans.cluster_centers_"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cluster Labels:\n",
      "[2 2 2 2 2 2 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 1\n",
      " 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n",
      " 2 2 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2\n",
      " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 2 2 2 2 2 2 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 1\n",
      " 1 1 1 1 1 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2\n",
      " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 0 0 0\n",
      " 0 0 0 2 2 2 2 2 2 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1]\n",
      "Cluster Centers:\n",
      "[[-8.26687967e-01  2.02278349e-01 -6.12372436e-01 -4.73803541e-01\n",
      "   1.33630621e-01 -1.85695338e-01  1.14707867e+00  6.88247202e-01\n",
      "  -1.06904497e+00  2.23606798e+00 -6.54653671e-01  3.42318766e-17\n",
      "   3.85494106e-17  3.26899002e-17  4.10165729e-17  3.91662011e-17\n",
      "   4.73386762e-17]\n",
      " [-2.87111260e-01 -1.21367009e-01 -3.62887369e-01  9.61964766e-01\n",
      "  -2.67261242e-01 -1.85695338e-01 -2.29415734e-01  2.54906371e-02\n",
      "  -1.06904497e+00 -4.47213595e-01  1.52752523e+00  3.36150860e-17\n",
      "   4.61564943e-17  3.00171410e-17  3.27926986e-17  2.53912118e-17\n",
      "   3.17133151e-17]\n",
      " [ 4.19840073e-01  5.05695871e-03  3.95490531e-01 -3.93041574e-01\n",
      "   1.08574880e-01  1.62483421e-01 -2.29415734e-01 -2.29415734e-01\n",
      "   9.35414347e-01 -4.47213595e-01 -6.54653671e-01  3.34223390e-17\n",
      "   4.00528376e-17  3.07238802e-17  2.59823027e-17  2.55197098e-17\n",
      "   3.12635720e-17]]\n"
     ]
    }
   ],
   "source": [
    "print(\"Cluster Labels:\")\n",
    "print(labels)\n",
    "print(\"Cluster Centers:\")\n",
    "print(centers)"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1600x900 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Use the correct variable name\n",
    "\n",
    "# Get cluster centers\n",
    "centers = kmeans.cluster_centers_\n",
    "\n",
    "# Plot cluster centers\n",
    "plt.scatter(centers[:, 0], centers[:, 1], c=\"black\", s=200, alpha=0.5)\n",
    "\n",
    "plt.xlabel(\"Feature 1\")\n",
    "plt.ylabel(\"Feature 2\")\n",
    "plt.title(\"K-Means Clustering\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['index', 'percentile_rank', 'percentile_category', 'region_0',\n",
      "       'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6',\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['index', 'percentile_rank', 'percentile_category', 'region_0',\n",
      "       'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6',\n",
      "      dtype='object')\n",
      "['region_0', 'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6']\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)  # Check column names\n",
    "region_column = [col for col in df.columns if \"region\" in col.lower()]\n",
    "print(region_column)  # See if a similar column exists\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "          region\n",
      "0  North America\n",
      "1  North America\n",
      "2  North America\n",
      "3  North America\n",
      "4  North America\n"
     ]
    }
   ],
   "source": [
    "# List of region column names\n",
    "region_columns = ['region_0', 'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6']\n",
    "\n",
    "# Define original region labels (Modify based on your data)\n",
    "\n",
    "# Map back to original category\n",
    "\n",
    "# Now, 'region' column exists\n",
    "print(df[[\"region\"]].head())  # Check if mapping worked\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "❌ Error: 'Cluster' column does not exist in DataFrame. Check K-Means execution.\n",
      "✅ K-Means successfully generated cluster labels.\n",
      "Cluster\n",
      "2    192\n",
      "1    108\n",
      "0     60\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# 🔹 Check if K-Means Clustering ran successfully\n",
    "if 'Cluster' not in df.columns:\n",
    "    print(\"❌ Error: 'Cluster' column does not exist in DataFrame. Check K-Means execution.\")\n",
    "\n",
    "# 🔹 Check if kmeans.labels_ exists\n",
    "if hasattr(kmeans, \"labels_\"):\n",
    "    print(\"✅ K-Means successfully generated cluster labels.\")\n",
    "else:\n",
    "\n",
    "# 🔹 Re-run clustering step to ensure clusters are assigned\n",
    "df[\"Cluster\"] = kmeans.labels_\n",
    "\n",
    "# 🔹 Verify if clusters were assigned correctly\n",
    "print(df[\"Cluster\"].value_counts())  # This should print count of each cluster\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 📌 Step 10: Visualize Clustered Data (Scatter Plot)\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.title(\"K-Means Clustering Visualization\")\n",
    "plt.xlabel(\"Feature 1 (Standardized)\")\n",
    "plt.ylabel(\"Feature 2 (Standardized)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 800x500 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# 📌 Step 11: Bar Graph of Cluster Counts\n",
    "# Bar Graph of Cluster Distribution\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.countplot(x=df[\"Cluster\"], palette=\"coolwarm\")\n",
    "plt.title(\"Cluster Distribution\")\n",
    "plt.xlabel(\"Cluster\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🔹 Summary of Clusters:\n",
      "Cluster\n",
      "2    192\n",
      "1    108\n",
      "0     60\n",
      "Name: count, dtype: int64\n",
      "Each data point is now assigned to a cluster for further analysis.\n"
     ]
    }
   ],
   "source": [
    "# 📌 Step 12: Conclusion\n",
    "print(\"\\n🔹 Summary of Clusters:\")\n",
    "print(df[\"Cluster\"].value_counts())\n",
    "print(\"Each data point is now assigned to a cluster for further analysis.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "📌 Findings: The model grouped countries into 3 clusters, showing AI skill trends."
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    " ️## Support Vector Machine (SVM)\n",
    "✅ Type: Classification\n",
    "✅ Purpose: Classifies AI skill levels into High, Medium, or Low categories.\n",
    "✅ Why SVM?\n",
    "\n",
    "*Works well when the data is not linearly separable\n",
    "*Good for small to medium datasets\n",
    "\n",
    " ## Logistic Regression\n",
    "✅ Type: Classification\n",
    "✅ Purpose: Attempts to predict whether a country falls into a high or low AI skill category.\n",
    "✅ Why Logistic Regression?\n",
    "\n",
    "*Simple and interpretable\n",
    "*Works best when features have a linear relationship"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 38.89%\n",
      "SVM Accuracy: 33.33%\n",
      "Logistic Regression Accuracy: 27.78%\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "models = {\n",
    "    \"Random Forest\": RandomForestClassifier(),\n",
    "    \"SVM\": SVC(),\n",
    "    \"Logistic Regression\": LogisticRegression()\n",
    "}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    accuracy = model.score(X_test, y_test) * 100\n",
    "    print(f\"{name} Accuracy: {accuracy:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "❌ Accuracy was too low (~33.33%), meaning SVM is not the best choice for this dataset.\n",
    "❌ Accuracy was too low (~27.78%), meaning logistic regression is not suitable for this dataset."
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "country                0\n",
      "iso3                   0\n",
      "region                 0\n",
      "incomegroup            0\n",
      "iso                    0\n",
      "percentile_rank        0\n",
      "percentile_category    0\n",
      "Cluster                0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(df.isnull().sum())  # See how many missing values each column has\n",
    "df = df.dropna()  # Remove rows with missing values (only if necessary)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "df = pd.get_dummies(df, columns=[\"region\", \"incomegroup\"], drop_first=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       'percentile_category', 'Cluster', 'region_1', 'region_2', 'region_3',\n",
      "       'region_4', 'region_5', 'region_6', 'incomegroup_1', 'incomegroup_2'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "X = df.drop(columns=[\"percentile_category\"])  # Remove target column\n",
    "y = df[\"percentile_category\"]  # Target column"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "X = df.drop(columns=[\"percentile_rank\"])  # Remove target column\n",
    "y = df[\"percentile_rank\"]  # Target column\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "float64\n",
      "[0.86440677 0.23728813 0.35593221 0.32203391 0.49152541 0.18644068\n",
      " 0.57627118 0.55932206 0.61016947 0.47457626]\n"
     ]
    }
   ],
   "source": [
    "print(y.dtype)\n",
    "print(y.unique()[:10])  # Show first 10 unique values\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "🔹 Supervised Learning Model\n",
    "## Random Forest Regressor (Best Performing Model)\n",
    "✅ Type: Regression\n",
    "✅ Purpose: Predicts the percentile rank of AI skills based on region, income group, and competency ID.\n",
    "✅ Why Random Forest?\n",
    "\n",
    "*Works well with structured data\n",
    "*Handles categorical and numerical features efficiently\n",
    "*Resistant to overfitting due to multiple decision trees"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 0.0066\n",
      "R² Score: 0.9263\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "# Split data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train Random Forest Regressor\n",
    "rf = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "rf.fit(X_train, y_train)\n",
    "\n",
    "# Make Predictions\n",
    "y_pred = rf.predict(X_test)\n",
    "\n",
    "# Evaluate the model\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(f\"Mean Squared Error: {mse:.4f}\")\n",
    "print(f\"R² Score: {r2:.4f}\")  # Closer to 1 means a better fit\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Get feature importances\n",
    "importances = rf.feature_importances_\n",
    "feature_names = X.columns\n",
    "\n",
    "# Sort and visualize\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.xlabel(\"Feature Importance\")\n",
    "plt.ylabel(\"Feature Name\")\n",
    "plt.title(\"Feature Importance in Random Forest Model\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8, 6))\n",
    "sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)\n",
    "plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color=\"red\", linestyle=\"dashed\")  # Perfect Prediction Line\n",
    "plt.xlabel(\"Actual Values\")\n",
    "plt.ylabel(\"Predicted Values\")\n",
    "plt.title(\"Actual vs Predicted Values\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "🔹 Why Use XGBoost?\n",
    "✅ Faster & More Efficient than Random Forest\n",
    "✅ Handles Missing Values Automatically\n",
    "✅ Prevents Overfitting using Regularization\n",
    "✅ Performs Well with Large Datasets"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "XGBoost Mean Squared Error: 0.0065\n",
      "XGBoost R² Score: 0.9275\n"
     ]
    }
   ],
   "source": [
    "from xgboost import XGBRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "# Train XGBoost model\n",
    "xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)\n",
    "xgb.fit(X_train, y_train)\n",
    "\n",
    "# Predict\n",
    "y_pred = xgb.predict(X_test)\n",
    "\n",
    "# Evaluate Performance\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(f\"XGBoost Mean Squared Error: {mse:.4f}\")\n",
    "print(f\"XGBoost R² Score: {r2:.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "\n",
    "🔹 What Are We Tuning?\n",
    "*n_estimators: Number of trees (50, 100, 200)\n",
    "*max_depth: Depth of each tree (3, 5, 10)\n",
    "*learning_rate: Step size (0.01, 0.1, 0.2)\n",
    "*subsample: Percentage of data used per tree (0.8, 1.0)"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 1.0}\n",
      "Optimized XGBoost Mean Squared Error: 0.0057\n",
      "Optimized XGBoost R² Score: 0.9370\n"
     ]
    }
   ],
   "source": [
    "\n",
    "    \"n_estimators\": [50, 100, 200],\n",
    "    \"max_depth\": [3, 5, 10],\n",
    "    \"learning_rate\": [0.01, 0.1, 0.2],\n",
    "    \"subsample\": [0.8, 1.0]\n",
    "}\n",
    "\n",
    "# Initialize XGBoost model\n",
    "xgb = XGBRegressor(random_state=42)\n",
    "\n",
    "\n",
    "# Best Parameters\n",
    "\n",
    "# Evaluate Best Model\n",
    "y_pred = best_xgb.predict(X_test)\n",
    "\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(f\"Optimized XGBoost Mean Squared Error: {mse:.4f}\")\n",
    "print(f\"Optimized XGBoost R² Score: {r2:.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "✅ XGBoost is a powerful boosting algorithm that performs well on structured data.\n",
    "✅ Compare R² Score of XGBoost with Random Forest to pick the best model."
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "## 🔹Add Feature Importance Visualization\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Get feature importances\n",
    "importances = best_xgb.feature_importances_\n",
    "feature_names = X.columns\n",
    "\n",
    "# Sort and visualize\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.xlabel(\"Feature Importance\")\n",
    "plt.ylabel(\"Feature Name\")\n",
    "plt.title(\"Feature Importance in Optimized XGBoost Model\")\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "📌 What Visualizations Are Included?\n",
    "\n",
    "1️⃣ Heatmap of Missing Data\n",
    "Shows missing values in the dataset.\n",
    "\n",
    "2️⃣ Boxplots\n",
    "Helps detect outliers in numerical features.\n",
    "\n",
    "3️⃣ Histograms\n",
    "Shows the distribution of numerical features.\n",
    "\n",
    "4️⃣ Correlation Heatmap\n",
    "Shows relationships between variables.\n",
    "\n",
    "5️⃣ K-Means Cluster Visualization\n",
    "Scatter plot showing data clusters.\n",
    "\n",
    "6️⃣ Bar Graph of Cluster Distribution\n",
    "Shows how many data points are in each cluster.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "# 📌 Conclusion & Key Findings\n",
    "\n",
    "### ✅ Model Performance\n",
    "- The **Random Forest Regressor** achieved **92.63% accuracy (R² Score)**, making it the best model for AI skill prediction.\n",
    "- **K-Means Clustering** successfully grouped countries based on AI skill trends.\n",
    "\n",
    "### 🔹 Future Improvements\n",
    "- Use **more features** like economic data or education levels for better predictions.\n",
    "- Experiment with **XGBoost** for improved performance.\n",
    "- Deploy this model as a **Streamlit Web App** for client usability.\n",
    "\n",
    "📌 This project can help businesses and policymakers **understand global AI skill trends** and **make data-driven decisions.**\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in c:\\users\\dell\\anaconda 3\\lib\\site-packages (1.30.0)\n",
      "Requirement already satisfied: xgboost in c:\\users\\dell\\anaconda 3\\lib\\site-packages (2.1.3)\n",
      "Requirement already satisfied: pandas in c:\\users\\dell\\anaconda 3\\lib\\site-packages (2.1.4)\n",
      "Requirement already satisfied: numpy in c:\\users\\dell\\anaconda 3\\lib\\site-packages (1.26.4)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\dell\\anaconda 3\\lib\\site-packages (1.2.2)\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (5.0.1)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (4.2.2)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: packaging<24,>=16.8 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (23.1)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (10.2.0)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (3.20.3)\n",
      "Requirement already satisfied: pyarrow>=6.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (14.0.2)\n",
      "Requirement already satisfied: python-dateutil<3,>=2.7.3 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (2.8.2)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (2.31.0)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (13.3.5)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (4.9.0)\n",
      "Requirement already satisfied: tzlocal<6,>=1.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (2.1)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (3.1.37)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (0.8.0)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (6.3.3)\n",
      "Requirement already satisfied: watchdog>=2.1.5 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from streamlit) (2.1.6)\n",
      "Requirement already satisfied: scipy in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from xgboost) (1.11.4)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from pandas) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: joblib>=1.1.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from scikit-learn) (1.2.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from scikit-learn) (2.2.0)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.3)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
      "Requirement already satisfied: toolz in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.7)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from python-dateutil<3,>=2.7.3->streamlit) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (1.26.18)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2024.2.2)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<5,>=3.0.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\dell\\anaconda 3\\lib\\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install streamlit xgboost pandas numpy scikit-learn"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "['xgboost_ai_skill_model.pkl']"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "# Save the trained model\n",
    "joblib.dump(best_xgb, \"xgboost_ai_skill_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting app.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile app.py\n",
    "\n",
    "print(\"Hello, this is a test file.\")"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Dell\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "print(os.getcwd()) "
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "region\n",
       "South America    0.622881\n",
       "Asia             0.603069\n",
       "North America    0.423729\n",
       "Africa           0.338983\n",
       "Oceania          0.209040\n",
       "Name: percentile_rank, dtype: float64"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"region\")[\"percentile_rank\"].mean().sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "region = st.selectbox(\n",
    "    \"Select Region\", \n",
    "    [\"South America\"],  # 👈 Only one option available\n",
    "    index=0  # 👈 Automatically selected\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "       'percentile_category', 'Cluster', 'region_1', 'region_2', 'region_3',\n",
      "       'region_4', 'region_5', 'region_6', 'incomegroup_1', 'incomegroup_2',\n",
      "       'region'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(df.columns)"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "income_group\n",
       "High      0.415725\n",
       "Name: percentile_rank, dtype: float64"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Identify which income group column is active for each row\n",
    "\n",
    "# Convert column names back to category labels\n",
    "df[\"income_group\"] = df[\"income_group\"].apply(lambda x: income_mapping[int(x.split(\"_\")[1])])\n",
    "\n",
    "# Now, check the best income group\n",
    "df.groupby(\"income_group\")[\"percentile_rank\"].mean().sort_values(ascending=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "income_group = st.selectbox(\n",
    "    \"Select Income Group\", \n",
    "    index=0  # 👈 Automatically selected\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "0    0.5\n",
       "1    0.5\n",
       "2    0.5\n",
       "3    0.5\n",
       "4    0.5\n",
       "5    0.5\n",
       "Name: percentile_rank, dtype: float64"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "    \"Competency ID (Skill Level)\", \n",
    "    0, 19,  # 👈 Keep range from 0 to 19\n",
    "    5  # 👈 Default value set to 5\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "if st.button(\"Predict AI Skill Rank\"):\n",
    "    prediction = model.predict(user_input)[0]\n",
    "    st.success(f\"Predicted AI Skill Rank: {prediction:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
     ]
    }
   ],
   "source": [
    "print(model.get_booster().feature_names)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "# Debugging - Print shape before prediction\n",
    "st.write(\"🔹 Checking Input Features for Debugging:\")\n",
    "st.write(f\"✅ Input Shape: {user_input.shape}\")  # Should be (1, 14)\n",
    "st.write(f\"✅ Feature Vector: {user_input}\")    # Should show 14 numbers"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "!notepad app.py"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "region = st.selectbox(\n",
    "    \"Select Region\", \n",
    "    [\"South America\"],  # 👈 Only one option available\n",
    "    index=0  # 👈 Ensures \"South America\" is selected by default\n",
    ")\n",
    "\n",
    "st.write(f\"🔹 Selected Region: {region}\")  # Debugging output"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "import numpy as np\n",
    "import streamlit as st\n",
    "import joblib\n",
    "\n",
    "# Load the trained XGBoost model\n",
    "model = joblib.load(\"xgboost_ai_skill_model.pkl\")\n",
    "\n",
    "st.title(\"AI Skill Ranking Prediction App\")\n",
    "\n",
    "# Fixing the region & income group selection\n",
    "region = \"South America\"  # Fixed to South America\n",
    "\n",
    "\n",
    "# Create a feature vector with 14 elements (all initialized to 0)\n",
    "feature_vector = np.zeros(14)\n",
    "\n",
    "# One-Hot Encode Region (Based on the 6 region features)\n",
    "region_mapping = {\n",
    "    \"South America\": 4  # Adjust this based on dataset encoding\n",
    "}\n",
    "feature_vector[6 + region_mapping[\"South America\"]] = 1  # Adjusting for 0-based index\n",
    "\n",
    "# One-Hot Encode Income Group (Based on 2 income group features)\n",
    "income_mapping = {\n",
    "}\n",
    "\n",
    "# Set Competency ID (Assuming it's at index 3)\n",
    "\n",
    "# Convert to NumPy array for prediction\n",
    "user_input = np.array([feature_vector])\n",
    "\n",
    "# Debugging Output\n",
    "st.write(\"🔹 Checking Input Features for Debugging:\")\n",
    "st.write(f\"✅ Input Shape: {user_input.shape}\")  # Should be (1, 14)\n",
    "st.write(f\"✅ Feature Vector: {user_input}\")    # Should show 14 numbers\n",
    "\n",
    "# Prediction Button\n",
    "if st.button(\"Predict AI Skill Rank\"):\n",
    "    prediction = model.predict(user_input)[0]\n",
    "    st.success(f\"Predicted AI Skill Rank: {prediction:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABVoAAAL4CAYAAACKtEyMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAuGklEQVR4nO3deZBV9Zn44bfpBZtFNiHgFqFIWAQRxLhE0WRQE40GR4EUURADaNAkZJIZx0QFSsU4OImD4Di4MZUwE4loRhNHxQAiaIIjkCjIuOASjSKyqWxNL78/UvZPbFQub0Njz/NUUdCnzz33PVDf7q4P555bVFNTUxMAAAAAAOy2Jg09AAAAAADAp53QCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAUklDD8Due+edd2Lbtm0NPQZQj0pKSqJNmzaxfv36qKysbOhxgHpkfUPjZX1D42V9Q+PVtGnT2H///ev1mELrp1hVVVVs3769occA9oDKykrrGxop6xsaL+sbGi/rGxqfkpL6z6JuHQAAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJBU0tADsPs2VRfFtiZNG3oMoB5trY7Y/PaGqKpuEjXWNzQq1jfsXFlRTRRXVTT0GAAAaULrp9icNTWxektNQ48B1LvKhh4A2GOsb/iwwZ2KoryhhwAAqAduHQAAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQFLJ7j5w8+bNMWbMmCgvL49//dd/jZKS/3+o+fPnx8033xyzZs362MfffffdsXjx4li7dm00a9YsevToEeecc0507tw5IiKWL18eEydOjKlTp0aHDh3qHGPatGmxZs2amDBhQp19J0yYEO3bt49LLrlkp88/YcKEWLFixQ7bSkpKok2bNnH00UfHsGHDoqysbHf+anbqkksuiZNOOimGDBlSb8cEAAAAAPYNux1aH3/88WjVqlVs2LAh/vCHP8QXv/jFgh5//fXXx/bt2+Piiy+Oz3zmM7Fx48a477774qqrrorrrrsuDj744E88xsiRI6O6unp3TyGOO+64GDlyZO3HW7dujT/+8Y8xY8aMqKqqim9961u7fWwAAAAA4P+O3b51wLx586JPnz7Ru3fvmDNnTkGPffXVV+PZZ5+NUaNGRa9evaJ9+/bRtWvX+N73vhctWrSI3/3ud7t0nGbNmkWLFi12Z/yIiCgrK4vWrVvX/urYsWOcdtppceKJJ8aiRYt2+7gAAAAAwP8tu3VF62uvvRbPP/98nHnmmbF169a4+eab47XXXtulq1AjIpo0+WvfXbp0aXTu3DmKiooiIqK4uDgmTpwYTZs23enj/vd//zeuvfbaOO200+Kb3/zmDrcOqE9lZWW1M0ZErF27NmbOnBlPP/10vPfee9G6desYMGBADB06NJo0aRLz58+PX/3qVzF48OCYPXt2rF27Nj772c/GyJEj4/Of/3yd42/dujWuu+662LRpU1x55ZXRqlWrep0fAAAAANi7diu0zps3L5o2bRp9+/aNqqqquPXWW2POnDk7vAz/4xx88MHRv3//uOuuu+KRRx6JI444Inr06BFHHHHETu/FGhHx/PPPx6RJk+L000+Pb3zjG7sz9ieqqqqKP/7xj7FgwYIYOHBg7faf/OQn0apVq/jxj38c5eXlsWTJkrjzzjuja9eucfTRR0dExLp162LOnDnxne98J0pKSuK2226LqVOnxr/8y7/UhuSIiIqKirj++utjy5YtcdVVV8X++++/R84FAAA+DYqKIkpLSxt6jJT336/ig+9bATQO1jc0XsXFxfV+zIK/UlRVVcVjjz0WRx11VO2Vp3379o0FCxbEsGHDPvJq1A/74Q9/GHPnzo2FCxfGggULYt68eVFUVBTHHXdcjBkzJpo1a1a776pVq+KWW26Jr33tazF48OBCR/5ICxcujN///ve1H1dUVET79u3jrLPOirPPPrt224ABA+LYY4+N9u3bR0TEV7/61fj1r38dr7zySm1oraqqitGjR8dhhx0WERFnn312TJ48OTZs2BBt2rSJiIjt27fvEFkztz0AAIDGoLi4ONof0K6hx6gX7//cDzQ+1jewKwoOrUuXLo0NGzbE8ccfX7vt+OOPj8WLF8eiRYviy1/+8i4dp0mTJjFw4MAYOHBgbN26NVauXBlPPPFEzJ8/P2pqauL73/9+7b433XRTVFZWfuTVrrvrqKOOivPOOy+qq6vjhRdeiH//93+P3r17x9lnn11btcvKyuIrX/lK/P73v4/f/va38eabb8Yrr7wS69evr/NGXAcddFDtn98PxZWVlbXbHnjggaisrIzDDz88mjdvXq/nAgAAn0ZVVVWxZs2ahh4jpaSkJNq0aRPr16/f4ed/4NPP+obGq2nTpvX+SvOCQ+u8efMiIuKnP/1pnc/NmTNnl0Lr4sWL4/XXX6+9anS//faLI488Mo488sho2bJlPPTQQzvsf84558SmTZtixowZccQRR9Tb/ySVl5dHx44dIyLiwAMPjLZt28bVV18dxcXFMWrUqIiI2LZtW4wfPz62bdsWxx13XAwYMCC6du0a48ePr3O8nb3kqaampvbPhx56aJx//vlx9dVXx5w5c+LUU0+tl/MAAIBPq5qav77yqzGorKxsNOcC7Mj6hsZnT9wSpKAjvvPOO7FkyZI4+eST42tf+9oOn3vggQdi7ty5sWrVqk88zttvvx2/+tWv4sQTT4wDDjhgh881a9YsWrduvcO2E044IVq3bh2LFy+O6dOnx2WXXVbI2LusV69eceaZZ8Z9990X/fv3jyOPPDKWLVsWq1atiunTp9fO9d5778WGDRsKPn7fvn2jZ8+eceaZZ8bMmTOjb9++tbcjAAAAAAA+vZoUsvOCBQuiuro6vv71r8ehhx66w6+//du/jSZNmsTDDz/8icf50pe+FJ/5zGdi4sSJ8dhjj8Xq1avj5ZdfjgcffDB+/etfxznnnFPnMWVlZTFmzJh46qmnYsGCBYWMXZAhQ4ZEp06dYvr06bF169Zo1+6v94t67LHHYs2aNbFy5cr4p3/6p6iqqtrt/80699xzo3Xr1nHLLbfU5+gAAAAAQAMpKLTOmzcvevfuvcO9SN/XoUOH+MIXvhCLFi2KzZs3f+xxysvL4+qrr47+/fvH3XffHX/3d38XV155ZTz++ONx6aWXxsknn7zTx/Xu3TtOPvnkmDFjxm5dUborysrK4qKLLoq1a9fGf/7nf0bXrl1j+PDh8cADD8S4ceNi2rRp0bNnz/jiF78YL7zwQuo5nnnmmXjkkUfq+QwAAAAAgL2tqOaDNxHlU+XOletj9Zaqhh4DAAB22+BORVFeva2hx0gpLS2N9u3bx5o1a9zDERoZ6xsar/Ly8np7H6j3FXRFKwAAAAAAdQmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkFTS0AOw+05pXxTbthc19BhAPSoqiiguLo6qqqqoqWnoaYD6ZH3DzpUVWRAAQOMgtH6KNW9SE02qtzX0GEA9Ki0tjfYHtIs1a9bE9u3bG3ocoB5Z3wAA0Li5dQAAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASSUNPQC7r7i4OEpLSxt6DKAelZSU7PA70HhY39B4Wd/QeFnf0HgVFxfX+zGLampqaur9qAAAAAAA/4e4dQAAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQFJJQw9AXdXV1XH33XfH7373u9i0aVN07949Ro0aFR07dtzp/u+++27ceeedsXTp0oiIOPbYY2PEiBGx33777c2xgV1Q6Pr+85//HL/4xS/i+eefjyZNmkTPnj1j+PDhccABB+zlyYFPUuj6/qCFCxfGlClTYurUqdGhQ4e9MC1QiELXd2VlZcyaNSseffTR2Lx5c3Tp0iVGjhwZhx122N4dHPhEha7vDRs2xIwZM+Lpp5+OiIjDDz88RowYEe3atdubYwMFmj17djz99NMxYcKEj9ynPvqaK1r3QbNnz445c+bERRddFNdcc00UFRXFpEmTorKycqf7//SnP43Vq1fHlVdeGT/4wQ/iT3/6U9x22217eWpgVxSyvt999924+uqro7y8PCZOnBg/+tGP4t13341rr702KioqGmB64OMU+v37fWvWrPF9G/Zxha7v2267LebOnRsXXXRR/OQnP4mWLVvGpEmTYvPmzXt5cuCTFLq+f/azn8XatWvjiiuuiCuuuCLWrl0bkydP3stTA4X47W9/G7NmzfrE/eqjrwmt+5jKysr4zW9+E4MHD45+/frFYYcdFuPGjYt169bFH/7whzr7P/fcc7F8+fIYO3ZsdOnSJXr16hVjxoyJxx57LNatW9cAZwB8lELX9+LFi2Pbtm0xduzYOOSQQ6JLly5x6aWXxuuvvx7PPfdcA5wB8FEKXd/vq66ujptuuim6dOmyF6cFClHo+n7rrbdi7ty5MXbs2OjXr18cdNBB8e1vfztKS0tj1apVDXAGwEcpdH1v2rQpnn322fj6178enTt3js6dO8fZZ58dq1atinfffbcBzgD4OOvWrYtJkybFL3/5yzjwwAM/dt/66mtC6z7m5Zdfji1btkSvXr1qtzVv3jw6d+4czz77bJ39n3322WjTpk0cdNBBtdsOP/zwiIhYuXLlnh8Y2GWFru/evXvH3//930dZWVmdz7333nt7dFagMIWu7/fde++9UVlZGYMGDdoLUwK7o9D1vWzZsmjevHkceeSRO+w/bdq0HY4BNLxC13dpaWk0bdq09rYgW7ZsiQULFkSnTp2iefPme3N0YBesWrUqmjdvHjfccEN07dr1Y/etr77mHq37mLVr10ZE1Ln/Yps2beLtt9/e6f4fvhdMSUlJtGzZcqf7Aw2n0PXdoUOHOvdqvPfee6O0tDR69Oix5wYFClbo+o6IeOGFF+L++++P6667zqtQYB9W6Pp+4403okOHDrF48eK49957Y926ddGlS5c4//zz4+CDD94rMwO7ptD1XVZWFt/+9rfj9ttvj5EjR9buO2HChGjSxHVssK/p379/9O/ff5f2ra++5ivBPmbbtm0R8dd/zA8qKyuL7du319m/oqIiSktL62wvLS3d6f5Awyl0fX/YAw88EA8//HAMGzYsWrVqtUdmBHZPoet769atMWXKlPjmN78ZnTp12iszArun0PW9ZcuWWL16dcyePTuGDRsWl112WRQXF8f48eNj48aNe2VmYNcUur5ramri1VdfjW7dusXEiRNj/Pjx0b59+7jhhhtiy5Yte2VmYM+or74mtO5j3n+J8IdvvF1RURFNmzbd6f47+wffvn37TvcHGk6h6/t9NTU18ctf/jJmzJgRgwYNijPOOGOPzgkUrtD1feedd0anTp3ilFNO2SvzAbuv0PVdUlISmzdvju9973vRp0+f6Nq1a4wbNy4iIh599NE9Pi+w6wpd34sWLYqHHnoovvOd70T37t2jZ8+ecdlll8WaNWti3rx5e2VmYM+or74mtO5j3n/JwodfQrh+/fpo27Ztnf3btWsX69ev32FbZWVlvPvuu3UueQYaVqHrO+Kv6/mmm26Ke++9N84777wYNmzYHp8TKFyh63vevHnxzDPPxPnnnx/nn39+TJo0KSIifvCDH8T06dP3/MDALit0fbdt2zaKi4t3uE1AWVlZdOjQId566609OyxQkELX98qVK+PAAw+M8vLy2m0tWrSIAw88MP7yl7/s2WGBPaq++prQuo/57Gc/G+Xl5bFixYrabZs2bYqXXnppp/dk7NGjR6xduzbefPPN2m3PPPNMRER069Ztzw8M7LJC13dExNSpU+OJJ56I7373u3HWWWftrVGBAhW6vqdMmRL//M//HJMnT47JkyfHxRdfHBERl19+eQwdOnSvzQ18skLXd8+ePaOqqipefPHF2m0VFRWxevXq6Nix416ZGdg1ha7vAw44IN54442oqKio3bZt27ZYvXq1WwHBp1x99TVvhrWPKS0tja985Ssxc+bM2H///aN9+/bxi1/8Itq1axfHHHNMVFdXxzvvvBPNmjWLsrKy+NznPhfdunWLG2+8MUaNGhVbt26NW2+9NU466aSPvEIOaBiFru/58+fH448/Huedd14cfvjhsWHDhtpjvb8PsG8odH1/OLZ88M043IMZ9i2Fru/u3btH7969Y+rUqTFmzJho2bJlzJo1K4qLi2PAgAENfTrABxS6vk866aS4//7748Ybb4yhQ4dGTU1N3HXXXVFaWhonn3xyQ58OUIA91deKampqavbg3OyG6urq+I//+I+YP39+VFRURI8ePeJb3/pW7cuNLr300hg7dmztF/KNGzfG7bffHkuXLo2ysrI47rjjYvjw4SIM7IMKWd/XXHNN/OlPf9rpcT74NQDYNxT6/fuDli9fHhMnToypU6dGhw4d9v7wwMcqdH1v2bIlZs6cGU888URUVFREt27d4oILLtjhdgLAvqHQ9f3aa6/FzJkz47nnnouioqLo3r17DB8+3Pdv2MdNmzYt1qxZExMmTIiI2GN9TWgFAAAAAEhyj1YAAAAAgCShFQAAAAAgSWgFAAAAAEgSWgEAAAAAkoRWAAAAAIAkoRUAAAAAIEloBQAAAABIEloBAAAAAJJKGnoAAACoL6+++mrcc889sXz58njvvfeiZcuW0aNHjxg0aFB07ty5dr8JEybs8PuHzZ8/P26++eaYOnVqdOjQIaZNmxYrVqyIadOmRUTEJZdcEj179oxLLrnkYx//YaWlpdG2bds46qij4hvf+Ebst99+uRPeiSFDhsS5554bQ4YMqfdjAwDw0YRWAAAahT//+c9xxRVXRNeuXWPkyJHRunXrWLt2bTz44INxxRVXxPjx4+Pzn//8Lh2rX79+cc0110SbNm1SM/3whz+M1q1b1368adOmWLZsWTzwwAOxYcOGGDduXOr4AADsO4RWAAAahd/85jfRokWL+NGPfhQlJf//x9yjjz46vv/978fs2bPj8ssv36Vj7b///rH//vunZzrssMOiQ4cOO2zr27dvbNy4MZ544om4+OKL98hVrQAA7H3u0QoAQKOwYcOGnW7fb7/9YsSIEXHcccd95GOXLVsWw4YNi5tvvjlqampi/vz5MWTIkHjrrbf2yKzNmjWrs23x4sVx1VVXxfDhw2PYsGExbty4ePDBB2s/v3z58hgyZEg8/fTTcc0118R5550Xo0ePjp///OdRVVX1kc911113xdChQ2Pu3Ll75FwAAPgrV7QCANAoHHXUUbF06dL48Y9/HF/60peiV69ecdBBB0VRUVEce+yxH/m4FStWxA033BDHH398XHzxxVFUVFRvM1VXV9dG0Jqamti8eXMsWbIkHn300fjCF75QezXrkiVL4oYbbojTTz89hgwZEtu2bYsHH3ww7rjjjujcuXN069at9phTpkyJ0047LQYNGhRPPfVU3H///dGxY8c45ZRT6jz/fffdF/fcc0+MHj06vvzlL9fbeQEAUJfQCgBAo3DqqafG+vXr47777os77rgjIiJatmwZffr0ia9+9avxuc99rs5jXnjhhbj++uvjmGOOibFjx0aTJvX7gq/vfve7dba1atUqTj311Bg6dGjtttdeey0GDBgQF1xwQe22bt26xYUXXhgrVqzYIbT+zd/8TZx77rkREdGrV6948skn46mnnqoTWufMmRMzZ86M0aNHx8CBA+v1vAAAqEtoBQCg0Rg6dGicccYZsWzZsnjmmWdi+fLlsXDhwli0aFGMGDEiTj/99Np933777Zg0aVJUV1fHqFGj6j2yRkT8wz/8Q7Rp0ya2b98e8+fPjwULFsSQIUPqRNGzzjorIiK2bt0ab775Zrzxxhvx4osvRkREZWXlDvt++A292rVrF9u2bdth21NPPRUvv/xydO/eXWQFANhLhFYAABqVFi1axAknnBAnnHBCRES89NJLMXXq1Jg5c2aceOKJ0bJly4iIeOutt6JPnz6xfPnymDVrVowYMaLeZzn00ENr3wyre/fuUVNTE7feemuUl5fXzhcR8c4778T06dPjySefjKKioujUqVPtVaw1NTU7HLNp06Y7fFxUVFRnn5deein69esXS5Ysif/5n/+J/v371/u5AQCwI2+GBQDAp966devioosu2ukbPnXu3DmGDh0a27dvj9WrV9duP+SQQ+If//Ef48wzz4z//u//jhdeeGGPz3nBBRdE27Zt4/bbb9/hzbumTJkSL774Ylx55ZXx85//PH72s5/FyJEjd/t5Bg4cGJdddll07949brvttti8eXM9TA8AwMcRWgEA+NRr3bp1NGnSJB566KGoqKio8/m//OUvUVpaGh07dqzd1rJlyyguLo5zzjkn2rdvH7fcckudl+nXt/Ly8hg+fHhs2rQpZs6cWbt95cqVccwxx0SvXr2itLQ0IiKWLl0aEXWvaN0VrVu3jqKiohg1alRs3Lhxh+cCAGDPEFoBAPjUa9KkSYwePTpeffXVuPzyy+Phhx+OFStWxNKlS2PGjBlx1113xeDBg6NFixZ1HltWVhYXXnhhvPrqq/Ff//Vfe3zW448/Pnr06BELFiyI5557LiIiunbtGgsXLowFCxbE8uXL45577olp06ZFUVFRnfuvFuLQQw+NM844Ix555JFYsWJFfZ0CAAA7IbQCANAo9OvXL6699to45JBD4t57741rr702brzxxnjllVdi3LhxMWjQoI98bN++fePYY4+Ne+65J15//fU9PuuFF14YRUVFcfvtt0d1dXVccskl0bVr17jjjjti8uTJ8eSTT8aYMWOiT58+8eyzz6aea/DgwXHAAQfEv/3bv+30al8AAOpHUc3uvBYJAAAAAIBarmgFAAAAAEgSWgEAAAAAkoRWAAAAAIAkoRUAAAAAIEloBQAAAABIEloBAAAAAJKEVgAAAACAJKEVAAAAACBJaAUAAAAASBJaAQAAAACShFYAAAAAgKT/B0H+mcd/b/O+AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1600x900 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Visualization code (e.g., bar chart for skill ranking)\n",
    "prediction = 0.56\n",
    "fig, ax = plt.subplots()\n",
    "ax.barh(['AI Skill Rank'], [prediction], color='skyblue')\n",
    "ax.set_xlim(0, 1)\n",
    "ax.set_xlabel('Skill Rank')\n",
    "plt.show()  # Use plt.show() to render in Jupyter Notebook\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABVoAAAL4CAYAAACKtEyMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAuGklEQVR4nO3deZBV9Zn44bfpBZtFNiHgFqFIWAQRxLhE0WRQE40GR4EUURADaNAkZJIZx0QFSsU4OImD4Di4MZUwE4loRhNHxQAiaIIjkCjIuOASjSKyqWxNL78/UvZPbFQub0Njz/NUUdCnzz33PVDf7q4P555bVFNTUxMAAAAAAOy2Jg09AAAAAADAp53QCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAUklDD8Due+edd2Lbtm0NPQZQj0pKSqJNmzaxfv36qKysbOhxgHpkfUPjZX1D42V9Q+PVtGnT2H///ev1mELrp1hVVVVs3769occA9oDKykrrGxop6xsaL+sbGi/rGxqfkpL6z6JuHQAAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJBU0tADsPs2VRfFtiZNG3oMoB5trY7Y/PaGqKpuEjXWNzQq1jfsXFlRTRRXVTT0GAAAaULrp9icNTWxektNQ48B1LvKhh4A2GOsb/iwwZ2KoryhhwAAqAduHQAAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQFLJ7j5w8+bNMWbMmCgvL49//dd/jZKS/3+o+fPnx8033xyzZs362MfffffdsXjx4li7dm00a9YsevToEeecc0507tw5IiKWL18eEydOjKlTp0aHDh3qHGPatGmxZs2amDBhQp19J0yYEO3bt49LLrlkp88/YcKEWLFixQ7bSkpKok2bNnH00UfHsGHDoqysbHf+anbqkksuiZNOOimGDBlSb8cEAAAAAPYNux1aH3/88WjVqlVs2LAh/vCHP8QXv/jFgh5//fXXx/bt2+Piiy+Oz3zmM7Fx48a477774qqrrorrrrsuDj744E88xsiRI6O6unp3TyGOO+64GDlyZO3HW7dujT/+8Y8xY8aMqKqqim9961u7fWwAAAAA4P+O3b51wLx586JPnz7Ru3fvmDNnTkGPffXVV+PZZ5+NUaNGRa9evaJ9+/bRtWvX+N73vhctWrSI3/3ud7t0nGbNmkWLFi12Z/yIiCgrK4vWrVvX/urYsWOcdtppceKJJ8aiRYt2+7gAAAAAwP8tu3VF62uvvRbPP/98nHnmmbF169a4+eab47XXXtulq1AjIpo0+WvfXbp0aXTu3DmKiooiIqK4uDgmTpwYTZs23enj/vd//zeuvfbaOO200+Kb3/zmDrcOqE9lZWW1M0ZErF27NmbOnBlPP/10vPfee9G6desYMGBADB06NJo0aRLz58+PX/3qVzF48OCYPXt2rF27Nj772c/GyJEj4/Of/3yd42/dujWuu+662LRpU1x55ZXRqlWrep0fAAAAANi7diu0zps3L5o2bRp9+/aNqqqquPXWW2POnDk7vAz/4xx88MHRv3//uOuuu+KRRx6JI444Inr06BFHHHHETu/FGhHx/PPPx6RJk+L000+Pb3zjG7sz9ieqqqqKP/7xj7FgwYIYOHBg7faf/OQn0apVq/jxj38c5eXlsWTJkrjzzjuja9eucfTRR0dExLp162LOnDnxne98J0pKSuK2226LqVOnxr/8y7/UhuSIiIqKirj++utjy5YtcdVVV8X++++/R84FAAA+DYqKIkpLSxt6jJT336/ig+9bATQO1jc0XsXFxfV+zIK/UlRVVcVjjz0WRx11VO2Vp3379o0FCxbEsGHDPvJq1A/74Q9/GHPnzo2FCxfGggULYt68eVFUVBTHHXdcjBkzJpo1a1a776pVq+KWW26Jr33tazF48OBCR/5ICxcujN///ve1H1dUVET79u3jrLPOirPPPrt224ABA+LYY4+N9u3bR0TEV7/61fj1r38dr7zySm1oraqqitGjR8dhhx0WERFnn312TJ48OTZs2BBt2rSJiIjt27fvEFkztz0AAIDGoLi4ONof0K6hx6gX7//cDzQ+1jewKwoOrUuXLo0NGzbE8ccfX7vt+OOPj8WLF8eiRYviy1/+8i4dp0mTJjFw4MAYOHBgbN26NVauXBlPPPFEzJ8/P2pqauL73/9+7b433XRTVFZWfuTVrrvrqKOOivPOOy+qq6vjhRdeiH//93+P3r17x9lnn11btcvKyuIrX/lK/P73v4/f/va38eabb8Yrr7wS69evr/NGXAcddFDtn98PxZWVlbXbHnjggaisrIzDDz88mjdvXq/nAgAAn0ZVVVWxZs2ahh4jpaSkJNq0aRPr16/f4ed/4NPP+obGq2nTpvX+SvOCQ+u8efMiIuKnP/1pnc/NmTNnl0Lr4sWL4/XXX6+9anS//faLI488Mo488sho2bJlPPTQQzvsf84558SmTZtixowZccQRR9Tb/ySVl5dHx44dIyLiwAMPjLZt28bVV18dxcXFMWrUqIiI2LZtW4wfPz62bdsWxx13XAwYMCC6du0a48ePr3O8nb3kqaampvbPhx56aJx//vlx9dVXx5w5c+LUU0+tl/MAAIBPq5qav77yqzGorKxsNOcC7Mj6hsZnT9wSpKAjvvPOO7FkyZI4+eST42tf+9oOn3vggQdi7ty5sWrVqk88zttvvx2/+tWv4sQTT4wDDjhgh881a9YsWrduvcO2E044IVq3bh2LFy+O6dOnx2WXXVbI2LusV69eceaZZ8Z9990X/fv3jyOPPDKWLVsWq1atiunTp9fO9d5778WGDRsKPn7fvn2jZ8+eceaZZ8bMmTOjb9++tbcjAAAAAAA+vZoUsvOCBQuiuro6vv71r8ehhx66w6+//du/jSZNmsTDDz/8icf50pe+FJ/5zGdi4sSJ8dhjj8Xq1avj5ZdfjgcffDB+/etfxznnnFPnMWVlZTFmzJh46qmnYsGCBYWMXZAhQ4ZEp06dYvr06bF169Zo1+6v94t67LHHYs2aNbFy5cr4p3/6p6iqqtrt/80699xzo3Xr1nHLLbfU5+gAAAAAQAMpKLTOmzcvevfuvcO9SN/XoUOH+MIXvhCLFi2KzZs3f+xxysvL4+qrr47+/fvH3XffHX/3d38XV155ZTz++ONx6aWXxsknn7zTx/Xu3TtOPvnkmDFjxm5dUborysrK4qKLLoq1a9fGf/7nf0bXrl1j+PDh8cADD8S4ceNi2rRp0bNnz/jiF78YL7zwQuo5nnnmmXjkkUfq+QwAAAAAgL2tqOaDNxHlU+XOletj9Zaqhh4DAAB22+BORVFeva2hx0gpLS2N9u3bx5o1a9zDERoZ6xsar/Ly8np7H6j3FXRFKwAAAAAAdQmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkFTS0AOw+05pXxTbthc19BhAPSoqiiguLo6qqqqoqWnoaYD6ZH3DzpUVWRAAQOMgtH6KNW9SE02qtzX0GEA9Ki0tjfYHtIs1a9bE9u3bG3ocoB5Z3wAA0Li5dQAAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASSUNPQC7r7i4OEpLSxt6DKAelZSU7PA70HhY39B4Wd/QeFnf0HgVFxfX+zGLampqaur9qAAAAAAA/4e4dQAAAAAAQJLQCgAAAACQJLQCAAAAACQJrQAAAAAASUIrAAAAAECS0AoAAAAAkCS0AgAAAAAkCa0AAAAAAElCKwAAAABAktAKAAAAAJAktAIAAAAAJAmtAAAAAABJQisAAAAAQFJJQw9AXdXV1XH33XfH7373u9i0aVN07949Ro0aFR07dtzp/u+++27ceeedsXTp0oiIOPbYY2PEiBGx33777c2xgV1Q6Pr+85//HL/4xS/i+eefjyZNmkTPnj1j+PDhccABB+zlyYFPUuj6/qCFCxfGlClTYurUqdGhQ4e9MC1QiELXd2VlZcyaNSseffTR2Lx5c3Tp0iVGjhwZhx122N4dHPhEha7vDRs2xIwZM+Lpp5+OiIjDDz88RowYEe3atdubYwMFmj17djz99NMxYcKEj9ynPvqaK1r3QbNnz445c+bERRddFNdcc00UFRXFpEmTorKycqf7//SnP43Vq1fHlVdeGT/4wQ/iT3/6U9x22217eWpgVxSyvt999924+uqro7y8PCZOnBg/+tGP4t13341rr702KioqGmB64OMU+v37fWvWrPF9G/Zxha7v2267LebOnRsXXXRR/OQnP4mWLVvGpEmTYvPmzXt5cuCTFLq+f/azn8XatWvjiiuuiCuuuCLWrl0bkydP3stTA4X47W9/G7NmzfrE/eqjrwmt+5jKysr4zW9+E4MHD45+/frFYYcdFuPGjYt169bFH/7whzr7P/fcc7F8+fIYO3ZsdOnSJXr16hVjxoyJxx57LNatW9cAZwB8lELX9+LFi2Pbtm0xduzYOOSQQ6JLly5x6aWXxuuvvx7PPfdcA5wB8FEKXd/vq66ujptuuim6dOmyF6cFClHo+n7rrbdi7ty5MXbs2OjXr18cdNBB8e1vfztKS0tj1apVDXAGwEcpdH1v2rQpnn322fj6178enTt3js6dO8fZZ58dq1atinfffbcBzgD4OOvWrYtJkybFL3/5yzjwwAM/dt/66mtC6z7m5Zdfji1btkSvXr1qtzVv3jw6d+4czz77bJ39n3322WjTpk0cdNBBtdsOP/zwiIhYuXLlnh8Y2GWFru/evXvH3//930dZWVmdz7333nt7dFagMIWu7/fde++9UVlZGYMGDdoLUwK7o9D1vWzZsmjevHkceeSRO+w/bdq0HY4BNLxC13dpaWk0bdq09rYgW7ZsiQULFkSnTp2iefPme3N0YBesWrUqmjdvHjfccEN07dr1Y/etr77mHq37mLVr10ZE1Ln/Yps2beLtt9/e6f4fvhdMSUlJtGzZcqf7Aw2n0PXdoUOHOvdqvPfee6O0tDR69Oix5wYFClbo+o6IeOGFF+L++++P6667zqtQYB9W6Pp+4403okOHDrF48eK49957Y926ddGlS5c4//zz4+CDD94rMwO7ptD1XVZWFt/+9rfj9ttvj5EjR9buO2HChGjSxHVssK/p379/9O/ff5f2ra++5ivBPmbbtm0R8dd/zA8qKyuL7du319m/oqIiSktL62wvLS3d6f5Awyl0fX/YAw88EA8//HAMGzYsWrVqtUdmBHZPoet769atMWXKlPjmN78ZnTp12iszArun0PW9ZcuWWL16dcyePTuGDRsWl112WRQXF8f48eNj48aNe2VmYNcUur5ramri1VdfjW7dusXEiRNj/Pjx0b59+7jhhhtiy5Yte2VmYM+or74mtO5j3n+J8IdvvF1RURFNmzbd6f47+wffvn37TvcHGk6h6/t9NTU18ctf/jJmzJgRgwYNijPOOGOPzgkUrtD1feedd0anTp3ilFNO2SvzAbuv0PVdUlISmzdvju9973vRp0+f6Nq1a4wbNy4iIh599NE9Pi+w6wpd34sWLYqHHnoovvOd70T37t2jZ8+ecdlll8WaNWti3rx5e2VmYM+or74mtO5j3n/JwodfQrh+/fpo27Ztnf3btWsX69ev32FbZWVlvPvuu3UueQYaVqHrO+Kv6/mmm26Ke++9N84777wYNmzYHp8TKFyh63vevHnxzDPPxPnnnx/nn39+TJo0KSIifvCDH8T06dP3/MDALit0fbdt2zaKi4t3uE1AWVlZdOjQId566609OyxQkELX98qVK+PAAw+M8vLy2m0tWrSIAw88MP7yl7/s2WGBPaq++prQuo/57Gc/G+Xl5bFixYrabZs2bYqXXnppp/dk7NGjR6xduzbefPPN2m3PPPNMRER069Ztzw8M7LJC13dExNSpU+OJJ56I7373u3HWWWftrVGBAhW6vqdMmRL//M//HJMnT47JkyfHxRdfHBERl19+eQwdOnSvzQ18skLXd8+ePaOqqipefPHF2m0VFRWxevXq6Nix416ZGdg1ha7vAw44IN54442oqKio3bZt27ZYvXq1WwHBp1x99TVvhrWPKS0tja985Ssxc+bM2H///aN9+/bxi1/8Itq1axfHHHNMVFdXxzvvvBPNmjWLsrKy+NznPhfdunWLG2+8MUaNGhVbt26NW2+9NU466aSPvEIOaBiFru/58+fH448/Huedd14cfvjhsWHDhtpjvb8PsG8odH1/OLZ88M043IMZ9i2Fru/u3btH7969Y+rUqTFmzJho2bJlzJo1K4qLi2PAgAENfTrABxS6vk866aS4//7748Ybb4yhQ4dGTU1N3HXXXVFaWhonn3xyQ58OUIA91deKampqavbg3OyG6urq+I//+I+YP39+VFRURI8ePeJb3/pW7cuNLr300hg7dmztF/KNGzfG7bffHkuXLo2ysrI47rjjYvjw4SIM7IMKWd/XXHNN/OlPf9rpcT74NQDYNxT6/fuDli9fHhMnToypU6dGhw4d9v7wwMcqdH1v2bIlZs6cGU888URUVFREt27d4oILLtjhdgLAvqHQ9f3aa6/FzJkz47nnnouioqLo3r17DB8+3Pdv2MdNmzYt1qxZExMmTIiI2GN9TWgFAAAAAEhyj1YAAAAAgCShFQAAAAAgSWgFAAAAAEgSWgEAAAAAkoRWAAAAAIAkoRUAAAAAIEloBQAAAABIEloBAAAAAJJKGnoAAACoL6+++mrcc889sXz58njvvfeiZcuW0aNHjxg0aFB07ty5dr8JEybs8PuHzZ8/P26++eaYOnVqdOjQIaZNmxYrVqyIadOmRUTEJZdcEj179oxLLrnkYx//YaWlpdG2bds46qij4hvf+Ebst99+uRPeiSFDhsS5554bQ4YMqfdjAwDw0YRWAAAahT//+c9xxRVXRNeuXWPkyJHRunXrWLt2bTz44INxxRVXxPjx4+Pzn//8Lh2rX79+cc0110SbNm1SM/3whz+M1q1b1368adOmWLZsWTzwwAOxYcOGGDduXOr4AADsO4RWAAAahd/85jfRokWL+NGPfhQlJf//x9yjjz46vv/978fs2bPj8ssv36Vj7b///rH//vunZzrssMOiQ4cOO2zr27dvbNy4MZ544om4+OKL98hVrQAA7H3u0QoAQKOwYcOGnW7fb7/9YsSIEXHcccd95GOXLVsWw4YNi5tvvjlqampi/vz5MWTIkHjrrbf2yKzNmjWrs23x4sVx1VVXxfDhw2PYsGExbty4ePDBB2s/v3z58hgyZEg8/fTTcc0118R5550Xo0ePjp///OdRVVX1kc911113xdChQ2Pu3Ll75FwAAPgrV7QCANAoHHXUUbF06dL48Y9/HF/60peiV69ecdBBB0VRUVEce+yxH/m4FStWxA033BDHH398XHzxxVFUVFRvM1VXV9dG0Jqamti8eXMsWbIkHn300fjCF75QezXrkiVL4oYbbojTTz89hgwZEtu2bYsHH3ww7rjjjujcuXN069at9phTpkyJ0047LQYNGhRPPfVU3H///dGxY8c45ZRT6jz/fffdF/fcc0+MHj06vvzlL9fbeQEAUJfQCgBAo3DqqafG+vXr47777os77rgjIiJatmwZffr0ia9+9avxuc99rs5jXnjhhbj++uvjmGOOibFjx0aTJvX7gq/vfve7dba1atUqTj311Bg6dGjtttdeey0GDBgQF1xwQe22bt26xYUXXhgrVqzYIbT+zd/8TZx77rkREdGrV6948skn46mnnqoTWufMmRMzZ86M0aNHx8CBA+v1vAAAqEtoBQCg0Rg6dGicccYZsWzZsnjmmWdi+fLlsXDhwli0aFGMGDEiTj/99Np933777Zg0aVJUV1fHqFGj6j2yRkT8wz/8Q7Rp0ya2b98e8+fPjwULFsSQIUPqRNGzzjorIiK2bt0ab775Zrzxxhvx4osvRkREZWXlDvt++A292rVrF9u2bdth21NPPRUvv/xydO/eXWQFANhLhFYAABqVFi1axAknnBAnnHBCRES89NJLMXXq1Jg5c2aceOKJ0bJly4iIeOutt6JPnz6xfPnymDVrVowYMaLeZzn00ENr3wyre/fuUVNTE7feemuUl5fXzhcR8c4778T06dPjySefjKKioujUqVPtVaw1NTU7HLNp06Y7fFxUVFRnn5deein69esXS5Ysif/5n/+J/v371/u5AQCwI2+GBQDAp966devioosu2ukbPnXu3DmGDh0a27dvj9WrV9duP+SQQ+If//Ef48wzz4z//u//jhdeeGGPz3nBBRdE27Zt4/bbb9/hzbumTJkSL774Ylx55ZXx85//PH72s5/FyJEjd/t5Bg4cGJdddll07949brvttti8eXM9TA8AwMcRWgEA+NRr3bp1NGnSJB566KGoqKio8/m//OUvUVpaGh07dqzd1rJlyyguLo5zzjkn2rdvH7fcckudl+nXt/Ly8hg+fHhs2rQpZs6cWbt95cqVccwxx0SvXr2itLQ0IiKWLl0aEXWvaN0VrVu3jqKiohg1alRs3Lhxh+cCAGDPEFoBAPjUa9KkSYwePTpeffXVuPzyy+Phhx+OFStWxNKlS2PGjBlx1113xeDBg6NFixZ1HltWVhYXXnhhvPrqq/Ff//Vfe3zW448/Pnr06BELFiyI5557LiIiunbtGgsXLowFCxbE8uXL45577olp06ZFUVFRnfuvFuLQQw+NM844Ix555JFYsWJFfZ0CAAA7IbQCANAo9OvXL6699to45JBD4t57741rr702brzxxnjllVdi3LhxMWjQoI98bN++fePYY4+Ne+65J15//fU9PuuFF14YRUVFcfvtt0d1dXVccskl0bVr17jjjjti8uTJ8eSTT8aYMWOiT58+8eyzz6aea/DgwXHAAQfEv/3bv+30al8AAOpHUc3uvBYJAAAAAIBarmgFAAAAAEgSWgEAAAAAkoRWAAAAAIAkoRUAAAAAIEloBQAAAABIEloBAAAAAJKEVgAAAACAJKEVAAAAACBJaAUAAAAASBJaAQAAAACShFYAAAAAgKT/B0H+mcd/b/O+AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1600x900 with 1 Axes>"
      ]
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Visualization code for Streamlit app\n",
    "prediction = 0.56\n",
    "fig, ax = plt.subplots()\n",
    "ax.barh(['AI Skill Rank'], [prediction], color='skyblue')\n",
    "ax.set_xlim(0, 1)\n",
    "ax.set_xlabel('Skill Rank')\n",
    "st.pyplot(fig)  # Display plot in Streamlit\n"
   ]
  },
  {
   "cell_type": "code",
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Assuming `prediction` is the predicted skill rank value\n",
    "prediction = 0.56  # For example\n",
    "\n",
    "# Create a bar chart for skill ranking\n",
    "st.subheader('Skill Ranking Prediction Visualization')\n",
    "fig, ax = plt.subplots()\n",
    "ax.barh(['AI Skill Rank'], [prediction], color='skyblue')\n",
    "ax.set_xlim(0, 1)  # Skill rank range from 0 to 1\n",
    "ax.set_xlabel('Skill Rank')\n",
    "st.pyplot(fig)\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
    "import numpy as np\n",
    "\n",
    "# Assuming y_true (true labels) and y_pred (predicted labels) are available\n",
    "y_true = [0.6, 0.8, 0.4, 0.7]  # Example true values\n",
    "y_pred = [0.56, 0.79, 0.39, 0.68]  # Example predicted values\n",
    "\n",
    "# Calculate MAE and RMSE\n",
    "mae = mean_absolute_error(y_true, y_pred)\n",
    "rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
    "\n",
    "# Display in Streamlit\n",
    "st.subheader('Model Performance')\n",
    "st.write(f\"Mean Absolute Error (MAE): {mae:.3f}\")\n",
    "st.write(f\"Root Mean Squared Error (RMSE): {rmse:.3f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "st.subheader(\"Enter Competency Features\")\n",
    "\n",
    "input_feature_vector = [[feature_1, feature_2, ...]]\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "st.title(\"AI Skill Ranking Prediction\")\n",
    "st.write(\"Enter the competency features below and click 'Predict' to receive the skill ranking.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "source": []
  }
 ],
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
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
