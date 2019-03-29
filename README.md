# Geographic Neighborhood Clustering

##### Author: Josh Kuppersmith

##### Date: March 29, 2019

##### Advisor: Dr. Pavlos Protopapas


![FinalMap](https://raw.githubusercontent.com/jkup11/Geographic-Neighborhood-Clustering/master/final_map.png)


This is the repository for Josh Kuppersmith's 2019 Senior Thesis in Applied Mathematics at Harvard University. 
The broad goal of the thesis is to successfully cluster a grid representation of a city into data-driven neighborhoods.

Included in this repository are data processing scripts and notebooks, an exploratory data analysis notebooks, and modeling notebooks and functions. In addition, the thesis writeup PDF is available to provide context to all functionality. To request access to the cleaned data used for this project, please reach out to Josh at joshkuppersmith@gmail.com

ABSTRACT: Open data initiatives in cities around the world have enabled new efforts to understand and improve urban areas through data analysis. In order to develop actionable insights to improve cities, it is important to isolate differences between geographic areas throughout the city. Neighborhoods are typically used as a unit for spatial separation, where each neighborhood is internally similar, and different from outside areas. As such, neighborhood analysis is key to developing an understanding of complex urban dynamics, yet current neighborhood boundaries do not always adequately reflect similar areas of cities. This thesis proposes a new clustering algorithm to automatically generate neighborhoods with highly similar internal data profiles. Using a grid-model of a city, this new method of clustering, called Geographic K-Means, incorporates data accumulated within grid cells and builds clumps of neighboring cells with similar data trends. This method is optimized using hyper-parameter tuning to improve an Earth Mover's Distance-based measure of within-neighborhood homogeneity. The optimization uses regularization to enforce smooth neighborhood boundaries, helping us find an optimal balance between data similarity and realistic contiguous neighborhoods. In order to build and test this algorithm, we used Chicago as a case study due to its abundance of data. By generating new Chicago neighborhood boundaries, and increasing within-neighborhood crime homogeneity, we are able to see the relationship between crime and neighborhoods, and better detect sharp boundaries between areas of the city.

With this new algorithm come a large number of possible applications. Some compelling applications include using voter data to generate optimal gerrymandering districts, using consumer data to better target customers with ads throughout a city, and using real estate data to find boundaries in housing markets in a city. I encourage other collaborators to continue testing this algorithm using new data, new cities, and new application goals. In addition, I hope that more work can be done to improve this model, and build off of these preliminary results. I encourage collaborators to reach out (joshkuppersmith@gmail.com) to discuss this project and possible future work. I hope that one day, geospatial machine learning expands into the mainstream, and tools like this begin to shape the way that we think about the world around us.
