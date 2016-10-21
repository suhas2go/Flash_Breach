
# Flash and Breaches
Solution to vehicle routing problem using machine learing techinques. Winning entry in Data Science Competition, Kriti 16, Intra-IIT Tech fest, IIT Guwahati.

### Problem Statement
Zoom, the arch-rival of Flash, and on a debatable note probably one of the few people who can travel faster than flash, has captured flash’s crew. The only way Zoom can travel between Earth 2 and our Earth is via many breaches that he created.

Flash finds out that his crew left him notes with the precise location of each of the breaches and a device which can close each one of them. The device functions on a highly volatile fuel and each breach requires fuel which is proportional to its size. The Flash can carry at most 986kg of weight with himself. The device weighs 11kg. Thus he would need to make several trips to the center. He can only save the crew if he can close all the breaches but one (well, he needs to get back to zoom) in the minimum time possible.

### Dataset 
[Breaches dataset](https://github.com/suhas2go/Flash_Breach/blob/master/data/Breach.csv)
### Data Exploration
* Distribution of points with increasing number of points

![](https://github.com/suhas2go/Flash_Breach/blob/master/res/img1.png)

* Distribution of fuel weights

![](https://github.com/suhas2go/Flash_Breach/blob/master/res/img2.png)

* Clusters(without constraints) generated using scikit-learn

![](https://github.com/suhas2go/Flash_Breach/blob/master/res/img3.png)

For more information about the data, check [this](https://github.com/suhas2go/Flash_Breach/blob/master/documentation/data exploration/Data Exploration.md) out.


### Method used
 * Modified K-Means Clustering Algorithm: a data point is assigned to the nearest cluster that is capable of accomodating the point.

For the detailed explanation about the exact steps taken to the solve the problem, please download and refer the [documentation](https://github.com/suhas2go/Flash_Breach/blob/master/documentation/Flash and Breach .html)
 
### Possible Improvements
 * Partitioning of the dataset to increase speed
 * Simulated Annealing to improve results
 
### References
> Clustering with Cluster Size Constraints Using a Modified k-means Algorithm
> -- <cite>[Nuwan Ganganath†, Chi-Tsun Cheng, and Chi K. Tse][1]</cite>

[1]:https://github.com/suhas2go/Flash_Breach/blob/master/res
