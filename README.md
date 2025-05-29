# Lagrangian Relaxation Algorithm for Network Revenue Management

This repository provides a public implementation of the Lagrangian Relaxation Algorithm for network revenue management, originally developed by Prof. Huseyin Topaloglu (2009).

Researchers in operations research often use this algorithm and its test dataset as benchmarks, but no open implementation was available until now. This code lets researchers test, compare, and build on the algorithm easily. I have tested this implementation on all instances in Prof. Topaloglu's dataset, and the results match those in his paper.

I developed this project as part of my research with Prof. Baris Ata on solving NRM problems using deep learning-based numerical methods. I hope this implementation supports reproducible research and further work in network revenue management.

For more details, see the [documentation](documentation/documentation.pdf).

## Resources

This project is based on [Topaloglu (2009)](https://people.orie.cornell.edu/huseyin/publications/revenue_man.pdf).
The paper and its dataset are available on [Prof. Huseyin Topaloglu's website](https://people.orie.cornell.edu/huseyin).
You can access the paper directly [here](https://people.orie.cornell.edu/huseyin/publications/revenue_man.pdf).
The dataset can be downloaded from [this page](https://people.orie.cornell.edu/huseyin/research/rm_datasets/rm_datasets.html).

## Installation

```bash
# Clone the repository
git clone https://github.com/steve-shao/LagrangianRelaxationNRM.git
cd LagrangianRelaxationNRM

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run all instances in the dataset
python run_all_instances.py
```

## Results

| Problem | Upper Bound (Huseyin) | Upper Bound (Our Impl.) | Mean Revenue (Huseyin) | Mean Revenue (Our Impl.) | Std (Our Impl., 1000 Samples) |
|---------|-----------------------|-------------------------|-------------------|---------------------|-----------------------|
| rm_200_4_1.0_4.0 | 20,439 | <font color="green">20,436</font> | 20,018 | <font color="green">20,049</font> | 31.31 |
| rm_200_4_1.0_8.0 | 33,305 | <font color="green">33,261</font> | 32,226 | <font color="green">32,821</font> | 62.70 |
| rm_200_4_1.2_4.0 | 18,938 | <font color="green">18,885</font> | 18,374 | <font color="green">18,510</font> | 28.49 |
| rm_200_4_1.2_8.0 | 31,737 | <font color="green">31,651</font> | 30,852 | <font color="green">31,271</font> | 64.73 |
| rm_200_4_1.6_4.0 | 16,600 | <font color="green">16,541</font> | 15,981 | <font color="green">16,186</font> | 27.72 |
| rm_200_4_1.6_8.0 | 29,413 | <font color="green">29,247</font> | 28,381 | <font color="green">28,978</font> | 63.80 |
| rm_200_5_1.0_4.0 | 21,298 | <font color="green">21,296</font> | 21,181 | <font color="red">20,973</font> | 34.79 |
| rm_200_5_1.0_8.0 | 34,393 | <font color="green">34,377</font> | 34,271 | <font color="red">34,068</font> | 69.42 |
| rm_200_5_1.2_4.0 | 20,184 | <font color="green">20,112</font> | 19,818 | <font color="red">19,677</font> | 33.06 |
| rm_200_5_1.2_8.0 | 33,165 | <font color="green">33,051</font> | 32,766 | <font color="red">32,620</font> | 68.80 |
| rm_200_5_1.6_4.0 | 17,704 | <font color="green">17,654</font> | 17,318 | <font color="red">17,218</font> | 30.93 |
| rm_200_5_1.6_8.0 | 30,594 | <font color="green">30,492</font> | 30,107 | <font color="red">29,980</font> | 66.84 |
| rm_200_6_1.0_4.0 | 21,128 | <font color="green">21,113</font> | 20,709 | <font color="green">20,729</font> | 33.13 |
| rm_200_6_1.0_8.0 | 34,178 | <font color="green">34,102</font> | 33,466 | <font color="green">33,664</font> | 66.86 |
| rm_200_6_1.2_4.0 | 19,649 | <font color="green">19,636</font> | 19,156 | <font color="green">19,165</font> | 31.25 |
| rm_200_6_1.2_8.0 | 32,566 | <font color="green">32,520</font> | 31,808 | <font color="green">31,993</font> | 67.36 |
| rm_200_6_1.6_4.0 | 17,304 | <font color="green">17,256</font> | 16,269 | <font color="green">16,837</font> | 30.08 |
| rm_200_6_1.6_8.0 | 30,170 | <font color="green">30,061</font> | 29,320 | <font color="green">29,599</font> | 65.70 |
| rm_200_8_1.0_4.0 | 18,975 | <font color="green">18,778</font> | 18,217 | <font color="green">18,268</font> | 31.10 |
| rm_200_8_1.0_8.0 | 30,490 | <font color="green">30,275</font> | 29,453 | <font color="green">29,716</font> | 66.44 |
| rm_200_8_1.2_4.0 | 17,472 | <font color="red">17,501</font> | 16,941 | <font color="red">16,915</font> | 29.44 |
| rm_200_8_1.2_8.0 | 28,908 | <font color="green">28,889</font> | 28,130 | <font color="green">28,236</font> | 61.56 |
| rm_200_8_1.6_4.0 | 15,295 | <font color="red">15,297</font> | 14,720 | <font color="green">14,764</font> | 27.34 |
| rm_200_8_1.6_8.0 | 26,661 | <font color="green">26,555</font> | 25,701 | <font color="green">25,988</font> | 63.75 |
| rm_600_4_1.0_4.0 | 30,995 | <font color="green">30,994</font> | 30,640 | <font color="red">30,575</font> | 49.25 |
| rm_600_4_1.0_8.0 | 50,444 | <font color="green">50,406</font> | 49,862 | <font color="green">49,872</font> | 107.31 |
| rm_600_4_1.2_4.0 | 28,668 | <font color="green">28,615</font> | 28,145 | <font color="green">28,167</font> | 44.65 |
| rm_600_4_1.2_8.0 | 48,054 | <font color="green">47,947</font> | 47,162 | <font color="green">47,541</font> | 101.67 |
| rm_600_4_1.6_4.0 | 25,148 | <font color="green">25,084</font> | 24,540 | <font color="green">24,596</font> | 43.50 |
| rm_600_4_1.6_8.0 | 44,555 | <font color="green">44,357</font> | 43,547 | <font color="green">44,003</font> | 102.95 |
| rm_600_5_1.0_4.0 | 32,254 | <font color="red">32,272</font> | 32,112 | <font color="red">31,775</font> | 56.33 |
| rm_600_5_1.0_8.0 | 52,071 | <font color="green">52,056</font> | 51,275 | <font color="green">51,668</font> | 118.57 |
| rm_600_5_1.2_4.0 | 30,004 | <font color="red">30,552</font> | 30,308 | <font color="red">30,100</font> | 51.57 |
| rm_600_5_1.2_8.0 | 50,282 | <font color="green">50,162</font> | 49,899 | <font color="red">49,629</font> | 114.59 |
| rm_600_5_1.6_4.0 | 26,936 | <font color="green">26,880</font> | 26,605 | <font color="red">26,441</font> | 44.95 |
| rm_600_5_1.6_8.0 | 46,497 | <font color="green">46,355</font> | 46,070 | <font color="red">45,858</font> | 107.20 |
| rm_600_6_1.0_4.0 | 25,541 | <font color="red">25,559</font> | 25,310 | <font color="red">25,044</font> | 47.32 |
| rm_600_6_1.0_8.0 | 41,412 | <font color="green">41,262</font> | 40,849 | <font color="red">40,753</font> | 102.27 |
| rm_600_6_1.2_4.0 | 23,687 | <font color="red">23,708</font> | 23,306 | <font color="red">23,191</font> | 42.77 |
| rm_600_6_1.2_8.0 | 39,307 | <font color="green">39,270</font> | 38,704 | <font color="green">38,799</font> | 100.42 |
| rm_600_6_1.6_4.0 | 20,817 | <font color="green">20,788</font> | 20,273 | <font color="red">20,229</font> | 41.46 |
| rm_600_6_1.6_8.0 | 36,381 | <font color="green">36,261</font> | 35,631 | <font color="green">35,867</font> | 101.19 |
| rm_600_8_1.0_4.0 | 22,960 | <font color="green">22,798</font> | 22,269 | <font color="red">22,206</font> | 44.93 |
| rm_600_8_1.0_8.0 | 36,933 | <font color="green">36,718</font> | 36,046 | <font color="green">36,071</font> | 95.73 |
| rm_600_8_1.2_4.0 | 21,102 | <font color="red">21,172</font> | 20,633 | <font color="red">20,431</font> | 39.54 |
| rm_600_8_1.2_8.0 | 34,831 | <font color="red">34,939</font> | 34,277 | <font color="red">34,168</font> | 88.63 |
| rm_600_8_1.6_4.0 | 18,500 | <font color="red">18,553</font> | 17,830 | <font color="green">17,888</font> | 35.95 |
| rm_600_8_1.6_8.0 | 32,247 | <font color="green">32,180</font> | 31,317 | <font color="green">31,411</font> | 90.19 |

## License

This project is licensed under the MIT License.

<br>