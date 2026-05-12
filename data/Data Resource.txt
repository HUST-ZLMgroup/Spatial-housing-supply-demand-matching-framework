# 说明数据来源和获取方式

# Data

本目录存放研究所用的所有原始数据。由于数据体量较大，所需数据请根据提供的URL自行下载
---

## 目录结构

```
data/
├── raw/
│   ├── buildings_raw/         # 建筑矢量数据（CMAB）
│   ├── nightlight/            # 夜间灯光栅格（NPP-VIIRS）
│   ├── poi/                   # 兴趣点数据（OSM）
│   ├── roads/                 # 道路网络数据（OSM）
│   ├── population/            # 人口栅格（LandScan）
│   ├── environment/           # 自然环境变量
│   ├── socioeconomic/         # 社会经济变量
│   └── correction/            # 省级住房面积修正系数表
└── processed/                 # 处理后中间数据（脚本自动生成，无需手动放置）
```

---

## 核心数据源

### 1. 建筑矢量数据（CMAB）

| 项目 | 说明 |
|------|------|
| 来源 | CMAB: A Multi-Attribute Building Dataset of China |
| 覆盖范围 | 中国约 3667 个城市区域，超过 3100 万栋建筑 |
| 主要属性 | 屋顶形态、高度、结构、功能、建筑风格、建成年代、建筑质量 |
| 下载地址 | https://figshare.com/articles/dataset/CMAB-The_World_s_First_National-Scale_Multi-Attribute_Building_Dataset/27992417/2 |
| 存放路径 | `data/raw/buildings_raw/<省份名>/` |


---

### 2. 夜间灯光数据（NPP-VIIRS）

| 项目 | 说明 |
|------|------|
| 来源 | Earth Observation Group（EOG） |
| 时间范围 | 2013–2023 年（逐年） |
| 空间分辨率 | 15 角秒（赤道附近约 500 m） |
| 产品说明 | 月度中位合成年度产品，过滤火灾、临时照明和背景噪声 |
| 下载地址 | https://eogdata.mines.edu/products/vnl/ |
| 存放路径 | `data/raw/nightlight/<year>/VNL_npp_<year>_*.tif` |

---

### 3. 兴趣点数据（POI / OSM）

| 项目 | 说明 |
|------|------|
| 来源 | OpenStreetMap contributors |
| 时间范围 | 2013–2023 年（逐年快照） |
| 下载平台 | Geofabrik：http://download.geofabrik.de/ |
| 存放路径 | `data/raw/poi/<year>/gis_osm_pois_*.shp` |

---

### 4. 道路网络数据（OSM）

| 项目 | 说明 |
|------|------|
| 来源 | OpenStreetMap |
| 时间范围 | 2013–2023 年（逐年快照） |
| 下载平台 | Geofabrik：http://download.geofabrik.de/ |
| 存放路径 | `data/raw/roads/<year>/gis_osm_roads_*.shp` |

---

### 5. 人口数据（LandScan / SSP）

| 项目 | 说明 |
|------|------|
| 历史（2013–2023）| LandScan Global，https://landscan.ornl.gov/ |
| 未来（2024–2050）| SSP 情景人口预测数据集，https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TLJ99B |
| 存放路径 | `data/raw/population/PopCount<year>.tif` |

---

## 解释变量数据源（42 类社会宏观变量）

### 自然环境变量

| 变量 | 全称 | 下载地址 |
|------|------|----------|
| ULA | Urban Land Area | https://doi.pangaea.de/10.1594/PANGAEA.905890 |
| ISA | Impervious Surface Area | https://figshare.com/articles/dataset/Global_fractional_urban_changes_at_1km_under_diverse_SSP-RCP_scenarios_throughout_2100/20391117/4 |
| WTD | Water Table Depth | https://geo.public.data.uu.nl/vault-globgm-cmip6-annual/ |
| HDS | Hydraulic Head |  https://geo.public.data.uu.nl/vault-globgm-cmip6-annual/ |
| LAI | Leaf Area Index | https://osf.io/9qz4k/overview |
| SAT | Near-surface Air Temperature | https://data.tpdc.ac.cn/en/data/40d649d6-d99e-45df-9814-c0115a109396 |
| MIT | Minimum Temperature | https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6 |
| MAT | Maximum Temperature | https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6 |
| SWS | Surface Wind Speed | https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6 |
| PR  | Precipitation | https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6 |
| RH  | Relative Humidity | https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6 |


### 经济与人口变量

| 变量 | 全称 | 下载地址 |
|------|------|----------|
| GDP | Gross Domestic Product | https://zenodo.org/records/5880037 |
| DI  | Disposable Income | https://springernature.figshare.com/articles/dataset/A_dataset_of_income_distribution_on_provincial_urban_and_rural_levels_for_China_from_2020_to_2100/27888801 |
| GC  | Gini Coefficient |  https://springernature.figshare.com/articles/dataset/A_dataset_of_income_distribution_on_provincial_urban_and_rural_levels_for_China_from_2020_to_2100/27888801 |
| POP | Population Count | https://landscan.ornl.gov/ |
| PGR | Population Growth Rate | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TLJ99B |
| VAP | Added Value of Primary Industry | https://www.scidb.cn/en/detail?dataSetId=73c1ddbd79e54638bd0ca2a6bd48e3ff |
| VAS | Added Value of Secondary Industry | https://www.scidb.cn/en/detail?dataSetId=73c1ddbd79e54638bd0ca2a6bd48e3ff |
| VAT | Added Value of Tertiary Industry | https://www.scidb.cn/en/detail?dataSetId=73c1ddbd79e54638bd0ca2a6bd48e3ff |

### 基础设施与公共服务变量

| 变量 | 全称 | 数据来源 |
|------|------|----------|
| RD  | Road Density | OSM / Geofabrik |
| DS  | Average Distance to Schools | OSM / Geofabrik |
| DH  | Average Distance to Hospitals | OSM / Geofabrik |
| DWB | Average Distance to Water Bodies | OSM / Geofabrik |
| DBS | Average Distance to Bus Stops | OSM / Geofabrik |
| DM  | Average Distance to Metro Stations | OSM / Geofabrik |
| DSM | Average Distance to Shopping Malls | OSM / Geofabrik |

### 城市社会发展指标

| 变量 | 全称 | 数据来源 |
|------|------|----------|
| WSR  | Water Supply Coverage Rate | EPSDATA |
| FAI  | Annual Fixed Asset Investment in Municipal Infrastructure | EPSDATA |
| STR  | Sewage Treatment Rate | EPSDATA |
| GCR  | Gas Coverage Rate | EPSDATA |
| WTR  | Domestic Waste Treatment Rate | EPSDATA |
| RMA  | Road Cleaning and Maintenance Area | EPSDATA |
| FPE  | Local Fiscal General Public Budget Expenditure | EPSDATA 中国城市数据库 |
| FPR  | Local Fiscal General Public Budget Revenue | EPSDATA |
| PI   | People Covered by Basic Pension Insurance (Urban Employees) | EPSDATA |
| UI   | People Covered by Unemployment Insurance | EPSDATA |
| ISD  | Industrial Sulfur Dioxide Emissions | EPSDATA |
| BLF  | Year-end Balance of Loans from Financial Institutions | EPSDATA |
| BDF  | Year-end Balance of Deposits in Financial Institutions | EPSDATA |
| RSC  | Total Retail Sales of Consumer Goods | EPSDATA |
| BMIE | People Covered by Basic Medical Insurance for Employees | EPSDATA |
| IED  | Number of Industrial Enterprises Above Designated Size | EPSDATA |

EPSDATA 平台入口：
- 中国城市数据库：https://olap.epsnet.com.cn/#/datas_home?cubeId=627
- 中国城乡建设数据库：https://olap.epsnet.com.cn/#/datas_home?cubeId=981


---

## 数据预处理说明

### 时间序列补齐

非逐年统计数据采用**线性插值**补齐年度时间序列。对于未来预测期（2024–2050 年）缺乏观测值的 OSM 类和城市统计类变量，采用**对数线性回归外推**：

```
y_{i,t} = β₀ + β₁ × ln(t - t_start + 1) + ε_{i,t}
```

其中 `t_start = 2013`，参数通过 OLS 拟合。对数时间变量能更好描述基础设施指标（如供水率、燃气覆盖率）在高覆盖水平下增速递减的特征。

### 缺失值处理

地理空间变量中的缺失值统一填充为 **-1**，标识该空间单元无可用观测值，避免模型训练时因缺失数据导致样本丢失。

### 空间基准

所有空间数据统一投影至 **WGS 84 / UTM Zone 49N（EPSG:32649）**，栅格数据重采样至 **10 km 网格**后进行空间叠加、距离计算和统计分析。