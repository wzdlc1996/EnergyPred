# README

## 目录设计:

请将数据文件移至根目录下, 应当有形式:

```
.
├── data_field
│   ├── dataTest.dat
│   ├── dataTrain.dat
│   ├── dataVali.dat
│   ├── valuTrain.dat
│   └── valuVali.dat
└── origin
    ├── pes-challenge-2019-fall.zip
    ├── README.pdf
    ├── sample.csv
    ├── test
    │   ├── asp_data
    │   │   ├── coord.dat
    │   │   └── type.dat
    │   ├── eth_data
    │   │   ├── coord.dat
    │   │   └── type.dat
    │   ├── mal_data
    │   │   ├── coord.dat
    │   │   └── type.dat
    │   ├── nap_data
    │   │   ├── coord.dat
    │   │   └── type.dat
    │   ├── sal_data
    │   │   ├── coord.dat
    │   │   └── type.dat
    │   ├── tol_data
    │   │   ├── coord.dat
    │   │   └── type.dat
    │   └── ura_data
    │       ├── coord.dat
    │       └── type.dat
    └── train
        ├── asp_data
        │   ├── coord.dat
        │   ├── ener.dat
        │   ├── force.dat
        │   └── type.dat
        ├── eth_data
        │   ├── coord.dat
        │   ├── ener.dat
        │   ├── force.dat
        │   └── type.dat
        ├── mal_data
        │   ├── coord.dat
        │   ├── ener.dat
        │   ├── force.dat
        │   └── type.dat
        ├── nap_data
        │   ├── coord.dat
        │   ├── ener.dat
        │   ├── force.dat
        │   └── type.dat
        ├── sal_data
        │   ├── coord.dat
        │   ├── ener.dat
        │   ├── force.dat
        │   └── type.dat
        ├── tol_data
        │   ├── coord.dat
        │   ├── ener.dat
        │   ├── force.dat
        │   └── type.dat
        └── ura_data
            ├── coord.dat
            ├── ener.dat
            ├── force.dat
            └── type.dat
```

其中 `origin/` 下为原始数据, `data_field/` 下为使用 `Mathematica` 程序得到的场的数据, 已经分好 training set 和 validation set

## Mathematica 文件说明

1.  `./molPlot.nb` : 用以按照给定原子构型绘制分子的代码
2.  `./PotentialField+nn/molToField.nb` : 将原始数据中的train部分转换成场数据, 并分离出validation set 的代码
3.  `./PotentialField+nn/testToField.nb` : 将测试数据集使用相同的算法转换为场数据的代码
4.  `PotentialField+nn/builtinPredictor.nb` : 使用Mathematica内置的机器学习程序包进行简单预测
