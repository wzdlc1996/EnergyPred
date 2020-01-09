# PES-Regression Project

This is the solution of one of the 2019 Kaggle Challenge: [PES-challenge](https://www.kaggle.com/c/pes-challenge-2019-fall)

Please download the data and arrange them as:

.
├── data
│   └── origin
│       ├── pes-challenge-2019-fall.zip
│       ├── README.pdf
│       ├── sample.csv
│       ├── test
│       │   ├── asp_data
│       │   │   ├── coord.dat
│       │   │   └── type.dat
│       │   ├── eth_data
│       │   │   ├── coord.dat
│       │   │   └── type.dat
│       │   ├── mal_data
│       │   │   ├── coord.dat
│       │   │   └── type.dat
│       │   ├── nap_data
│       │   │   ├── coord.dat
│       │   │   └── type.dat
│       │   ├── sal_data
│       │   │   ├── coord.dat
│       │   │   └── type.dat
│       │   ├── tol_data
│       │   │   ├── coord.dat
│       │   │   └── type.dat
│       │   └── ura_data
│       │       ├── coord.dat
│       │       └── type.dat
│       └── train
│           ├── asp_data
│           │   ├── coord.dat
│           │   ├── ener.dat
│           │   ├── force.dat
│           │   └── type.dat
│           ├── eth_data
│           │   ├── coord.dat
│           │   ├── ener.dat
│           │   ├── force.dat
│           │   └── type.dat
│           ├── mal_data
│           │   ├── coord.dat
│           │   ├── ener.dat
│           │   ├── force.dat
│           │   └── type.dat
│           ├── nap_data
│           │   ├── coord.dat
│           │   ├── ener.dat
│           │   ├── force.dat
│           │   └── type.dat
│           ├── sal_data
│           │   ├── coord.dat
│           │   ├── ener.dat
│           │   ├── force.dat
│           │   └── type.dat
│           ├── tol_data
│           │   ├── coord.dat
│           │   ├── ener.dat
│           │   ├── force.dat
│           │   └── type.dat
│           └── ura_data
│               ├── coord.dat
│               ├── ener.dat
│               ├── force.dat
│               └── type.dat
├── data_field
│   ├── dataTrain.dat
│   ├── dataVali.dat
│   ├── valuTrain.dat
│   └── valuVali.dat
├── mma
│   ├── molPlot.nb
│   ├── molPreHand.nb
│   └── molToField.nb
└── README.md

Where `./data/origin/` stores all extracted data file from Kaggle official site.
