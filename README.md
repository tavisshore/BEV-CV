<div align="center">    
 
# BEV-CV: Birds-Eye-View Transform for Cross-View Geo-Localisation
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

![new_neural](https://github.com/user-attachments/assets/84215eee-31b0-4ca6-871e-cacf329c6347#gh-light-mode-only)
![spagbol_diagram](https://github.com/user-attachments/assets/4f3d921f-c24b-409f-a2e7-c9669a4d98a6#gh-dark-mode-only)


</div>
 
## 📓 Description 
Cross-view image matching for geo-localisation is a challenging problem due to the significant visual difference between aerial and ground-level viewpoints. The method provides localisation capabilities from geo-referenced images, eliminating the need for external devices or costly equipment. This enhances the capacity of agents to autonomously determine their position, navigate, and operate effectively in GNSS-denied environments. Current research employs a variety of techniques to reduce the domain gap such as applying polar transforms to aerial images or synthesising between perspectives. However, these approaches generally rely on having a 360° field of view, limiting real-world feasibility. We propose BEV-CV, an approach introducing two key novelties with a focus on improving the real-world viability of cross-view geo-localisation. Firstly bringing ground-level images into a semantic Birds-Eye-View before matching embeddings, allowing for direct comparison with aerial image representations. Secondly, we adapt datasets into application realistic format - limited Field-of-View images aligned to vehicle direction. BEV-CV achieves state-of-the-art recall accuracies, improving Top-1 rates of 70° crops of CVUSA and CVACT by 23% and 24% respectively. Also decreasing computational requirements by reducing floating point operations to below previous works, and decreasing embedding dimensionality by 33% - together allowing for faster localisation capabilities. 


---
## 🧰 SpaGBOL: Benchmarking

🚧 Under Construction

```

```

```

```

```

```


## BEV-CV: Evaluation


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

### 🦜 [SpaGBOL: Spatial-Graph-Based Orientated Localisation](https://github.com/tavisshore/SpaGBOL)
[![Paper](http://img.shields.io/badge/ArXiv-2312.15363-B31B1B.svg)](https://arxiv.org/abs/2312.15363)
[![Conference](http://img.shields.io/badge/IROS-2024-4b44ce.svg)](https://wacv2025.thecvf.com/)
[![Project Page](http://img.shields.io/badge/Project-Page-green)](https://tavisshore.co.uk/bevcv/)
[![GitHub](https://img.shields.io/badge/GitHub-BEVCV-%23121011.svg?logo=github&logoColor=white)](https://github.com/tavisshore/bevcv)
```
@InProceedings{Shore_2025_WACV,
    author    = {Shore, Tavis and Mendez, Oscar and Hadfield, Simon},
    title     = {SpaGBOL: Spatial-Graph-Based Orientated Localisation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025}
}
```

