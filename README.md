<div align="center">    
 
# 🦜🌍 BEV-CV: Birds-Eye-View Transform for Cross-View Geo-Localisation 📡🗺️
<p align="middle">
 <a href="https://tavisshore.co.uk/">Tavis Shore</a>
 <a href="https://personalpages.surrey.ac.uk/s.hadfield/biography.html">Simon Hadfield</a>
 <a href="https://cvssp.org/Personal/OscarMendez/index.html">Oscar Mendez</a>
</p>
<p align="middle">
 <a href="https://www.surrey.ac.uk/centre-vision-speech-signal-processing">Centre for Vision, Speech, and Signal Processing (CVSSP)</a>
</p>
<p align="middle">
 <a>University of Surrey, Guildford, GU2 7XH, United Kingdom </a>
</p>

[![Paper](http://img.shields.io/badge/ArXiv-2312.15363-B31B1B.svg)](https://arxiv.org/abs/2312.15363)
[![Conference](http://img.shields.io/badge/IROS-2024-4b44ce.svg)](http://iros2024-abudhabi.org/)
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/bevcv/)
[![License](https://img.shields.io/badge/license-MIT-blue)]()

![bevcv_dark_mode](https://github.com/user-attachments/assets/159bceea-220d-4b12-a5aa-feb3baea7d4e#gh-dark-mode-only)
![bevcv](https://github.com/user-attachments/assets/08c60b11-fd12-46ab-aec6-75a53b4d7b8d#gh-light-mode-only)

</div>
 
## 📓 Description 
Cross-view image matching for geo-localisation is a challenging problem due to the significant visual difference between aerial and ground-level viewpoints. The method provides localisation capabilities from geo-referenced images, eliminating the need for external devices or costly equipment. This enhances the capacity of agents to autonomously determine their position, navigate, and operate effectively in GNSS-denied environments. Current research employs a variety of techniques to reduce the domain gap such as applying polar transforms to aerial images or synthesising between perspectives. However, these approaches generally rely on having a 360° field of view, limiting real-world feasibility. We propose BEV-CV, an approach introducing two key novelties with a focus on improving the real-world viability of cross-view geo-localisation. Firstly bringing ground-level images into a semantic Birds-Eye-View before matching embeddings, allowing for direct comparison with aerial image representations. Secondly, we adapt datasets into application realistic format - limited Field-of-View images aligned to vehicle direction. BEV-CV achieves state-of-the-art recall accuracies, improving Top-1 rates of 70° crops of CVUSA and CVACT by 23% and 24% respectively. Also decreasing computational requirements by reducing floating point operations to below previous works, and decreasing embedding dimensionality by 33% - together allowing for faster localisation capabilities. 


---
## 🧰 BEV-CV: Benchmarking

🚧 Under Construction

#### 🏭 Data Pre-Processing
```

```

#### Submodule Pretraining
```

```

#### BEV-CV Training
```

```

#### BEV-CV Evaluation
```

```

## BEV-CV: Benchmark Results

<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow">Model</th>
    <th class="tg-c3ow">Orientation<br>Aware</th>
    <th class="tg-c3ow">R@1</th>
    <th class="tg-c3ow">R@5</th>
    <th class="tg-c3ow">R@10</th>
    <th class="tg-c3ow">R@1%</th>
    <th class="tg-c3ow">R@1</th>
    <th class="tg-c3ow">R@5</th>
    <th class="tg-c3ow">R@10</th>
    <th class="tg-c3ow">R@1\%</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow" colspan="4">CVUSA 90</td>
    <td class="tg-c3ow" colspan="4">CVUSA 70</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CVM</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">2.76</td>
    <td class="tg-c3ow">10.11</td>
    <td class="tg-c3ow">16.74</td>
    <td class="tg-c3ow">55.49</td>
    <td class="tg-c3ow">2.62</td>
    <td class="tg-c3ow">9.30</td>
    <td class="tg-c3ow">15.06</td>
    <td class="tg-c3ow">21.77</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CVFT</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">4.80</td>
    <td class="tg-c3ow">14.84</td>
    <td class="tg-c3ow">23.18</td>
    <td class="tg-c3ow">61.23</td>
    <td class="tg-c3ow">3.79</td>
    <td class="tg-c3ow">12.44</td>
    <td class="tg-c3ow">19.33</td>
    <td class="tg-c3ow">55.56</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DSM</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">16.19</td>
    <td class="tg-c3ow">31.44</td>
    <td class="tg-c3ow">39.85</td>
    <td class="tg-c3ow">71.13</td>
    <td class="tg-c3ow">8.78</td>
    <td class="tg-c3ow">19.90</td>
    <td class="tg-c3ow">27.30</td>
    <td class="tg-c3ow">61.20</td>
  </tr>
  <tr>
    <td class="tg-c3ow">L2LTR</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">26.92</td>
    <td class="tg-c3ow">50.49</td>
    <td class="tg-c3ow">60.41</td>
    <td class="tg-c3ow">86.88</td>
    <td class="tg-c3ow">13.95</td>
    <td class="tg-c3ow">33.07</td>
    <td class="tg-c3ow">43.86</td>
    <td class="tg-c3ow">77.65</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TransGeo</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">30.12</td>
    <td class="tg-c3ow">54.18</td>
    <td class="tg-c3ow">63.96</td>
    <td class="tg-c3ow">89.18</td>
    <td class="tg-c3ow">16.43</td>
    <td class="tg-c3ow">37.28</td>
    <td class="tg-c3ow">48.02</td>
    <td class="tg-c3ow">80.75</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GeoDTR</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">18.81</td>
    <td class="tg-c3ow">43.36</td>
    <td class="tg-c3ow">57.94</td>
    <td class="tg-c3ow">88.14</td>
    <td class="tg-c3ow">14.84</td>
    <td class="tg-c3ow">38.03</td>
    <td class="tg-c3ow">51.27</td>
    <td class="tg-c3ow">88.17</td>
  </tr>
  <tr>
    <td class="tg-c3ow">BEV-CV</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">15.17</td>
    <td class="tg-c3ow">33.91</td>
    <td class="tg-c3ow">45.33</td>
    <td class="tg-c3ow">82.53</td>
    <td class="tg-c3ow">14.03</td>
    <td class="tg-c3ow">32.32</td>
    <td class="tg-c3ow">43.25</td>
    <td class="tg-c3ow">81.48</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GAL</td>
    <td class="tg-c3ow">≈</td>
    <td class="tg-c3ow">22.54</td>
    <td class="tg-c3ow">44.36</td>
    <td class="tg-c3ow">54.17</td>
    <td class="tg-c3ow">84.59</td>
    <td class="tg-c3ow">15.20</td>
    <td class="tg-c3ow">32.86</td>
    <td class="tg-c3ow">42.06</td>
    <td class="tg-c3ow">75.21</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DSM</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-7btt">33.66</td>
    <td class="tg-c3ow">51.70</td>
    <td class="tg-c3ow">59.68</td>
    <td class="tg-c3ow">82.46</td>
    <td class="tg-c3ow">20.88</td>
    <td class="tg-c3ow">36.99</td>
    <td class="tg-c3ow">44.70</td>
    <td class="tg-c3ow">71.10</td>
  </tr>
  <tr>
    <td class="tg-c3ow">L2LTR</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">25.21</td>
    <td class="tg-c3ow">51.90</td>
    <td class="tg-c3ow">63.54</td>
    <td class="tg-c3ow">91.16</td>
    <td class="tg-c3ow">22.20</td>
    <td class="tg-c3ow">46.71</td>
    <td class="tg-c3ow">58.99</td>
    <td class="tg-c3ow">89.37</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TransGeo</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">21.96</td>
    <td class="tg-c3ow">45.35</td>
    <td class="tg-c3ow">56.49</td>
    <td class="tg-c3ow">86.80</td>
    <td class="tg-c3ow">17.27</td>
    <td class="tg-c3ow">38.95</td>
    <td class="tg-c3ow">49.44</td>
    <td class="tg-c3ow">81.34</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GeoDTR</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">15.21</td>
    <td class="tg-c3ow">39.32</td>
    <td class="tg-c3ow">52.27</td>
    <td class="tg-c3ow">88.72</td>
    <td class="tg-c3ow">14.00</td>
    <td class="tg-c3ow">35.28</td>
    <td class="tg-c3ow">47.77</td>
    <td class="tg-c3ow">86.39</td>
  </tr>
  <tr>
    <td class="tg-c3ow">BEV-CV</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">32.11</td>
    <td class="tg-7btt">58.36</td>
    <td class="tg-7btt">69.06</td>
    <td class="tg-7btt">92.99</td>
    <td class="tg-7btt">27.40</td>
    <td class="tg-7btt">52.94</td>
    <td class="tg-7btt">64.47</td>
    <td class="tg-7btt">90.94</td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow" colspan="4">CVACT 90</td>
    <td class="tg-c3ow" colspan="4">CVACT 70</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CVM</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">1.47</td>
    <td class="tg-c3ow">5.70</td>
    <td class="tg-c3ow">9.64</td>
    <td class="tg-c3ow">38.05</td>
    <td class="tg-c3ow">1.24</td>
    <td class="tg-c3ow">4.98</td>
    <td class="tg-c3ow">8.42</td>
    <td class="tg-c3ow">34.74</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CVFT</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">1.85</td>
    <td class="tg-c3ow">6.28</td>
    <td class="tg-c3ow">10.54</td>
    <td class="tg-c3ow">39.25</td>
    <td class="tg-c3ow">1.49</td>
    <td class="tg-c3ow">5.13</td>
    <td class="tg-c3ow">8.19</td>
    <td class="tg-c3ow">34.59</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DSM</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">18.11</td>
    <td class="tg-c3ow">33.34</td>
    <td class="tg-c3ow">40.94</td>
    <td class="tg-c3ow">68.65</td>
    <td class="tg-c3ow">8.29</td>
    <td class="tg-c3ow">20.72</td>
    <td class="tg-c3ow">27.13</td>
    <td class="tg-c3ow">57.08</td>
  </tr>
  <tr>
    <td class="tg-c3ow">L2LTR</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">13.07</td>
    <td class="tg-c3ow">30.38</td>
    <td class="tg-c3ow">41.00</td>
    <td class="tg-c3ow">76.07</td>
    <td class="tg-c3ow">6.67</td>
    <td class="tg-c3ow">15.94</td>
    <td class="tg-c3ow">23.45</td>
    <td class="tg-c3ow">49.37</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TransGeo</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">10.75</td>
    <td class="tg-c3ow">28.22</td>
    <td class="tg-c3ow">37.51</td>
    <td class="tg-c3ow">70.15</td>
    <td class="tg-c3ow">7.01</td>
    <td class="tg-c3ow">19.44</td>
    <td class="tg-c3ow">27.50</td>
    <td class="tg-c3ow">62.19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GeoDTR</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">26.53</td>
    <td class="tg-c3ow">53.26</td>
    <td class="tg-c3ow">64.59</td>
    <td class="tg-c3ow">91.13</td>
    <td class="tg-c3ow">16.87</td>
    <td class="tg-c3ow">40.22</td>
    <td class="tg-c3ow">53.13</td>
    <td class="tg-c3ow">87.92</td>
  </tr>
  <tr>
    <td class="tg-c3ow">BEV-CV</td>
    <td class="tg-c3ow">❌</td>
    <td class="tg-c3ow">4.14</td>
    <td class="tg-c3ow">14.46</td>
    <td class="tg-c3ow">22.64</td>
    <td class="tg-c3ow">61.18</td>
    <td class="tg-c3ow">3.92</td>
    <td class="tg-c3ow">13.50</td>
    <td class="tg-c3ow">20.53</td>
    <td class="tg-c3ow">59.34</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GAL</td>
    <td class="tg-c3ow">≈</td>
    <td class="tg-c3ow">26.05</td>
    <td class="tg-c3ow">49.23</td>
    <td class="tg-c3ow">59.26</td>
    <td class="tg-c3ow">85.60</td>
    <td class="tg-c3ow">14.17</td>
    <td class="tg-c3ow">32.96</td>
    <td class="tg-c3ow">43.24</td>
    <td class="tg-c3ow">77.49</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DSM</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">31.17</td>
    <td class="tg-c3ow">51.44</td>
    <td class="tg-c3ow">60.05</td>
    <td class="tg-c3ow">82.90</td>
    <td class="tg-c3ow">18.44</td>
    <td class="tg-c3ow">35.87</td>
    <td class="tg-c3ow">44.39</td>
    <td class="tg-c3ow">71.97</td>
  </tr>
  <tr>
    <td class="tg-c3ow">L2LTR</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">33.62</td>
    <td class="tg-c3ow">46.28</td>
    <td class="tg-c3ow">58.21</td>
    <td class="tg-c3ow">78.62</td>
    <td class="tg-c3ow">28.65</td>
    <td class="tg-c3ow">53.59</td>
    <td class="tg-c3ow">65.02</td>
    <td class="tg-c3ow">90.48</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TransGeo</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">28.16</td>
    <td class="tg-c3ow">34.44</td>
    <td class="tg-c3ow">41.54</td>
    <td class="tg-c3ow">67.15</td>
    <td class="tg-c3ow">24.05</td>
    <td class="tg-c3ow">42.68</td>
    <td class="tg-c3ow">55.47</td>
    <td class="tg-c3ow">80.72</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GeoDTR</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-c3ow">26.76</td>
    <td class="tg-c3ow">53.65</td>
    <td class="tg-c3ow">65.35</td>
    <td class="tg-c3ow">92.12</td>
    <td class="tg-c3ow">15.38</td>
    <td class="tg-c3ow">37.09</td>
    <td class="tg-c3ow">49.40</td>
    <td class="tg-c3ow">86.38</td>
  </tr>
  <tr>
    <td class="tg-7btt">BEV-CV</td>
    <td class="tg-c3ow">✅</td>
    <td class="tg-7btt">45.79</td>
    <td class="tg-7btt">75.85</td>
    <td class="tg-7btt">83.97</td>
    <td class="tg-7btt">96.76</td>
    <td class="tg-7btt">37.85</td>
    <td class="tg-7btt">69.00</td>
    <td class="tg-7btt">78.52</td>
    <td class="tg-7btt">95.03</td>
  </tr>
</tbody></table>






## ✒️ Citation   
```
@INPROCEEDINGS{bevcv,
    author={Shore, Tavis and Hadfield, Simon and Mendez, Oscar },
    booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
    title={BEV-CV: Birds-Eye-View Transform for Cross-View Geo-Localisation}, 
    year={2024},
    pages={11047-11054},
}
```
## 📗 Related Works

### 🍝 [SpaGBOL: Spatial-Graph-Based Orientated Localisation](https://github.com/tavisshore/SpaGBOL)
[![Paper](http://img.shields.io/badge/ArXiv-2312.15363-B31B1B.svg)](https://arxiv.org/abs/2312.15363)
[![Conference](http://img.shields.io/badge/IROS-2024-4b44ce.svg)](https://wacv2025.thecvf.com/)
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/spagbol/)
[![GitHub](https://img.shields.io/badge/GitHub-SpaGBOL-%23121011.svg?logo=github&logoColor=white)](https://github.com/tavisshore/spagbol)
```
@InProceedings{Shore_2025_WACV,
    author    = {Shore, Tavis and Mendez, Oscar and Hadfield, Simon},
    title     = {SpaGBOL: Spatial-Graph-Based Orientated Localisation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025}
}
```

