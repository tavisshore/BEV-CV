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

```

```

```

```

```

```

## BEV-CV: Evaluation
<table><thead>
  <tr>
    <th>Model</th>
    <th>Orientation<br>Aware</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1%</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1\%</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td colspan="4">CVUSA 90</td>
    <td colspan="4">CVUSA 70</td>
  </tr>
  <tr>
    <td>CVM</td>
    <td>\xmark</td>
    <td>2.76</td>
    <td>10.11</td>
    <td>16.74</td>
    <td>55.49</td>
    <td>2.62</td>
    <td>9.30</td>
    <td>15.06</td>
    <td>21.77</td>
  </tr>
  <tr>
    <td>CVFT</td>
    <td>\xmark</td>
    <td>4.80</td>
    <td>14.84</td>
    <td>23.18</td>
    <td>61.23</td>
    <td>3.79</td>
    <td>12.44</td>
    <td>19.33</td>
    <td>55.56</td>
  </tr>
  <tr>
    <td>DSM</td>
    <td>\xmark</td>
    <td>16.19</td>
    <td>31.44</td>
    <td>39.85</td>
    <td>71.13</td>
    <td>8.78</td>
    <td>19.90</td>
    <td>27.30</td>
    <td>61.20</td>
  </tr>
  <tr>
    <td>L2LTR</td>
    <td>\xmark</td>
    <td>26.92</td>
    <td>50.49</td>
    <td>60.41</td>
    <td>86.88</td>
    <td>13.95</td>
    <td>33.07</td>
    <td>43.86</td>
    <td>77.65</td>
  </tr>
  <tr>
    <td>TransGeo</td>
    <td>\xmark</td>
    <td>30.12</td>
    <td>54.18</td>
    <td>63.96</td>
    <td>89.18</td>
    <td>16.43</td>
    <td>37.28</td>
    <td>48.02</td>
    <td>80.75</td>
  </tr>
  <tr>
    <td>GeoDTR</td>
    <td>\xmark</td>
    <td>18.81</td>
    <td>43.36</td>
    <td>57.94</td>
    <td>88.14</td>
    <td>14.84</td>
    <td>38.03</td>
    <td>51.27</td>
    <td>88.17</td>
  </tr>
  <tr>
    <td>BEV-CV</td>
    <td>\xmark</td>
    <td>15.17</td>
    <td>33.91</td>
    <td>45.33</td>
    <td>82.53</td>
    <td>14.03</td>
    <td>32.32</td>
    <td>43.25</td>
    <td>81.48</td>
  </tr>
  <tr>
    <td>GAL</td>
    <td>approx</td>
    <td>22.54</td>
    <td>44.36</td>
    <td>54.17</td>
    <td>84.59</td>
    <td>15.20</td>
    <td>32.86</td>
    <td>42.06</td>
    <td>75.21</td>
  </tr>
  <tr>
    <td>DSM</td>
    <td>\cmark</td>
    <td>33.66</td>
    <td>51.70</td>
    <td>59.68</td>
    <td>82.46</td>
    <td>20.88</td>
    <td>36.99</td>
    <td>44.70</td>
    <td>71.10</td>
  </tr>
  <tr>
    <td>L2LTR</td>
    <td>\cmark</td>
    <td>25.21</td>
    <td>51.90</td>
    <td>63.54</td>
    <td>91.16</td>
    <td>22.20</td>
    <td>46.71</td>
    <td>58.99</td>
    <td>89.37</td>
  </tr>
  <tr>
    <td>TransGeo</td>
    <td>\cmark</td>
    <td>21.96</td>
    <td>45.35</td>
    <td>56.49</td>
    <td>86.80</td>
    <td>17.27</td>
    <td>38.95</td>
    <td>49.44</td>
    <td>81.34</td>
  </tr>
  <tr>
    <td>GeoDTR</td>
    <td>\cmark</td>
    <td>15.21</td>
    <td>39.32</td>
    <td>52.27</td>
    <td>88.72</td>
    <td>14.00</td>
    <td>35.28</td>
    <td>47.77</td>
    <td>86.39</td>
  </tr>
  <tr>
    <td>BEV-CV</td>
    <td>\cmark</td>
    <td>32.11</td>
    <td>58.36</td>
    <td>69.06</td>
    <td>92.99</td>
    <td>27.40</td>
    <td>52.94</td>
    <td>64.47</td>
    <td>90.94</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td colspan="4">CVACT 90</td>
    <td colspan="4">CVACT 70</td>
  </tr>
  <tr>
    <td>CVM</td>
    <td>\xmark</td>
    <td>1.47</td>
    <td>5.70</td>
    <td>9.64</td>
    <td>38.05</td>
    <td>1.24</td>
    <td>4.98</td>
    <td>8.42</td>
    <td>34.74</td>
  </tr>
  <tr>
    <td>CVFT</td>
    <td>\xmark</td>
    <td>1.85</td>
    <td>6.28</td>
    <td>10.54</td>
    <td>39.25</td>
    <td>1.49</td>
    <td>5.13</td>
    <td>8.19</td>
    <td>34.59</td>
  </tr>
  <tr>
    <td>DSM</td>
    <td>\xmark</td>
    <td>18.11</td>
    <td>33.34</td>
    <td>40.94</td>
    <td>68.65</td>
    <td>8.29</td>
    <td>20.72</td>
    <td>27.13</td>
    <td>57.08</td>
  </tr>
  <tr>
    <td>L2LTR</td>
    <td>\xmark</td>
    <td>13.07</td>
    <td>30.38</td>
    <td>41.00</td>
    <td>76.07</td>
    <td>6.67</td>
    <td>15.94</td>
    <td>23.45</td>
    <td>49.37</td>
  </tr>
  <tr>
    <td>TransGeo</td>
    <td>\xmark</td>
    <td>10.75</td>
    <td>28.22</td>
    <td>37.51</td>
    <td>70.15</td>
    <td>7.01</td>
    <td>19.44</td>
    <td>27.50</td>
    <td>62.19</td>
  </tr>
  <tr>
    <td>GeoDTR</td>
    <td>\xmark</td>
    <td>26.53</td>
    <td>53.26</td>
    <td>64.59</td>
    <td>91.13</td>
    <td>16.87</td>
    <td>40.22</td>
    <td>53.13</td>
    <td>87.92</td>
  </tr>
  <tr>
    <td>BEV-CV</td>
    <td>\xmark</td>
    <td>4.14</td>
    <td>14.46</td>
    <td>22.64</td>
    <td>61.18</td>
    <td>3.92</td>
    <td>13.50</td>
    <td>20.53</td>
    <td>59.34</td>
  </tr>
  <tr>
    <td>GAL</td>
    <td>approx</td>
    <td>26.05</td>
    <td>49.23</td>
    <td>59.26</td>
    <td>85.60</td>
    <td>14.17</td>
    <td>32.96</td>
    <td>43.24</td>
    <td>77.49</td>
  </tr>
  <tr>
    <td>DSM</td>
    <td>\cmark</td>
    <td>31.17</td>
    <td>51.44</td>
    <td>60.05</td>
    <td>82.90</td>
    <td>18.44</td>
    <td>35.87</td>
    <td>44.39</td>
    <td>71.97</td>
  </tr>
  <tr>
    <td>L2LTR</td>
    <td>\cmark</td>
    <td>33.62</td>
    <td>46.28</td>
    <td>58.21</td>
    <td>78.62</td>
    <td>28.65</td>
    <td>53.59</td>
    <td>65.02</td>
    <td>90.48</td>
  </tr>
  <tr>
    <td>TransGeo</td>
    <td>\cmark</td>
    <td>28.16</td>
    <td>34.44</td>
    <td>41.54</td>
    <td>67.15</td>
    <td>24.05</td>
    <td>42.68</td>
    <td>55.47</td>
    <td>80.72</td>
  </tr>
  <tr>
    <td>GeoDTR</td>
    <td>\cmark</td>
    <td>26.76</td>
    <td>53.65</td>
    <td>65.35</td>
    <td>92.12</td>
    <td>15.38</td>
    <td>37.09</td>
    <td>49.40</td>
    <td>86.38</td>
  </tr>
  <tr>
    <td>BEV-CV</td>
    <td>\cmark</td>
    <td>45.79</td>
    <td>75.85</td>
    <td>83.97</td>
    <td>96.76</td>
    <td>37.85</td>
    <td>69.00</td>
    <td>78.52</td>
    <td>95.03</td>
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

