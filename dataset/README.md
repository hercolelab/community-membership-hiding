# Dataset Overview

This folder contains the dataset used for CMH experiments. It includes .txt data collected from: 

### [1. KONECT](http://konect.cc/)
```bibtex
@misc{konect, 
    url={http://konect.cc/}, 
    journal={Konect.cc}, 
    year={2023}
}
```

### [2. Network Repository](https://networkrepository.com/index.php)
```bibtex
@misc{nets, 
    url={http://www-personal.umich.edu/~mejn/netdata/}, 
    journal={Umich.edu}, 
    year={2013} 
}
```

### [3. SNAP Large Networks](https://snap.stanford.edu/data/index.html)
```bibtex
@misc{snapnets,
    author       = {Jure Leskovec and Andrej Krevl},
    title        = {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
    howpublished = {\url{http://snap.stanford.edu/data}},
    month        = jun,
    year         = 2014
}
```

# Networks

## Zachary Karate Club  
[kar](http://konect.cc/networks/ucidata-zachary/): This classic network, collected by Wayne Zachary in 1977, maps relationships among 34 university karate club members. Each node represents a member, and each edge signifies a connection. The club later split into two groups after a dispute between instructors.  

- **Nodes:** 34  
- **Edges:** 78  

## David Copperfield  

[words](http://konect.cc/networks/adjnoun_adjacency/): This undirected network represents noun and adjective adjacencies in *David Copperfield* by Dickens. Nodes are words (nouns or adjectives), and edges connect words appearing consecutively. The network is not bipartite, meaning words of the same type can be linked.  

- **Nodes:** 112  
- **Edges:** 425

## Wikipedia Voting Network  

[vote](https://networkrepository.com/soc-wiki-Vote.php): This dataset includes all Wikipedia voting data from its inception until January 2008. Nodes represent Wikipedia users, and directed edges indicate that one user voted for another.  

- **Nodes:** 889  
- **Edges:** 2,914

## Western U.S. Power Grid  

[pow](http://konect.cc/networks/opsahl-powergrid/): This undirected network represents the power grid of the Western United States. Nodes are generators, transformers, or substations, while edges represent power supply lines.  

- **Nodes:** 4,941  
- **Edges:** 6,594

## Facebook Friendship Network  

[fb-75](https://networkrepository.com/socfb-American75.php): This social network represents friendships on Facebook, where nodes are individuals and edges denote friendship connections.  

- **Nodes:** 6,386  
- **Edges:** 217,662  

## Condense Matter Collaboration Network 

[cond-mat](https://snap.stanford.edu/data/ca-CondMat.html): This undirected network represents scientific collaborations in the Condensed Matter Physics category of arXiv from January 1993 to April 2003. Nodes are authors, and edges indicate co-authorship. A paper with multiple authors forms a fully connected subgraph. 

- **Nodes:** 23,133  
- **Edges:** 93,497
