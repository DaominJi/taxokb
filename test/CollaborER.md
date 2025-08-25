# CollaborER: A Self-supervised Entity Resolution Framework Using Multi-features Collaboration 

Congcong Ge, Pengfei Wang, Lu Chen, Xiaoze Liu, Baihua Zheng, Yunjun Gao, Member, IEEE


#### Abstract

Entity Resolution (ER) aims to identify whether two tuples refer to the same real-world entity and is well-known to be labor-intensive. It is a prerequisite to anomaly detection, as comparing the attribute values of two matched tuples from two different datasets provides one effective way to detect anomalies. Existing ER approaches, due to insufficient feature discovery or error-prone inherent characteristics, are not able to achieve a stable performance. In this paper, we present CollaborER, a self-supervised entity resolution framework via multi-features collaboration. It is capable of (i) obtaining reliable ER results with zero human annotations and (ii) discovering adequate tuples' features in a fault-tolerant manner. CollaborER consists of two phases, i.e., automatic label generation (ALG) and collaborative ER training (CERT). In the first phase, ALG is proposed to generate a set of positive tuple pairs and a set of negative tuple pairs. ALG guarantees the high quality of the generated tuples, and hence ensures the training quality of the subsequent CERT. In the second phase, CERT is introduced to learn the matching signals by discovering graph features and sentence features of tuples collaboratively. Extensive experimental results over eight real-world ER benchmarks show that CollaborER outperforms all the existing unsupervised ER approaches and is comparable or even superior to the state-of-the-art supervised ER methods.


Index Terms-Entity Resolution, Sentence Feature, Graph Feature, Self-supervised, Anomaly Detection

## 1 INTRODUCTION

Due to widespread data quality issues [11], anomaly detection has received tremendous attention in diverse domains. It aims to find anomalous data in a dataset. Many studies [2], [53] focus on anomaly detection based on the information provided by a single dataset. Differently, we would like to highlight that the anomaly detection problem can be facilitated by considering the information captured by different sources, since it is common that different data sources can provide information about the same real-world entity [7]. Given two tuples from different relational datasets that refer to the same entity in real life, if they contain the same attribute but have contradictory values in the cell of the attribute, at least one of the values is an anomaly.

Example 1. Figure 1 depicts two sampled tables, each of which contains three tuples about products gathered from Amazon and Google, respectively. In this figure, we assume the matched tuples (connected by tick-marked lines) that refer to the same real-world entity have been perfectly identified. Since $e_{1}^{\prime}$ and $e_{1}$ are matched tuples, it is expected that the two tuples share the same values for a given attribute. However, after comparing the attribute values of $e_{1}$ with the attribute values of $e_{1}^{\prime}$, we can easily spot an anomalous value w.r.t. the attribute Title of tuple $e_{1}^{\prime}$, which might be caused by data extraction errors. Specifically, the value "aspyr media inc" of $e_{1}^{\prime}$,

- C. Ge, L. Chen, X. Liu, and Y. Gao (Corresponding Author) are with the College of Computer Science, Zhejiang University, Hangzhou 310027, China, E-mail:[gcc, luchen, xiaoze, gaoqf]@2ju.edu.cn.
- P. Wang is with the School of Software, Zhejiang University, Hangzhou, China, E-mail:wangpf@2ju.edu.cn.
- B. Zheng is with the School of Computing and Information Systems, Singapore Management University, Singapore 178902, Singapore, E-mail: bkzheng@smu.edu.sg.
![img-0.jpeg](img-0.jpeg)

Fig. 1. Example of using ER for anomaly detection
corresponding to the attribute Manufacturer, is wrongly extracted into the cell of attribute Title.

Considering that it is impractical to assume all the matched tuples are known beforehand, in this paper, we focus on Entity Resolution (ER) [7], which aims to identify whether a tuple from one relational dataset and another tuple from a different relational dataset refer to the same real-world entity. It is worth noting that reliable ER results are a prerequisite to ensure the quality of multi-sourcebased anomaly detection. This is because anomaly detection on top of false-aligned tuples is meaningless.

Early ER approaches require rules [12], [37] or crowdsourcing [14], [29], [44], which are impractical for matching real-world entities with literal or symbolic heterogeneity. Recently, embedding has become an increasingly powerful tool to encode heterogeneous entities into a unified semantic vector space, giving birth to various embedding-based ER techniques. Current embedding-based solutions to ER mostly rely on either sentence features or graph features. The

former [10], [13], [18], [22], [25], [32], [51] treats each tuple as a sentence and learns the tuple's embedding according to the contextual information contained in the sentence. The latter [5], [23] first constructs graphs to represent tuples and then learns matching signals of tuples based on the graph structure. Despite the considerable ER performance on several benchmarks, identifying tuples referring to the same real-world entity, however, is still a challenging endeavor. The challenges are mainly two-fold, as listed below.
Challenge I: Labor-intensive annotations for generating prematched tuples. Embedding-based ER can achieve considerable results but typically requires a large number of labeled tuple pairs. The annotating process is labor-intensive and hence restricts the scope of its applications in real-world ER scenarios. Although several ER approaches [5], [46], [49] have tried to perform ER in an unsupervised way without any annotation, their ER performance is far from satisfactory due to the error-sensitive nature. A large body of research [3], [15], [26] has indicated that unsupervised methods can be easily misled/fooled or attacked since they do not include any supervision signal. Therefore, even slight erroneous data may lead to wrong results. For example, ZeroER [46], the state-of-the-art unsupervised ER method, achieves poor performance on dirty datasets, as confirmed in the experiments to be presented in Section 6.3. Since realworld ER datasets usually incorporate various errors, it is challenging to apply unsupervised methods to solve realworld ER tasks directly.
Challenge II: Insufficient feature discovery of the tuples for ER. Based on our preliminary study, neither sentence-based nor graph-based approaches is able to discover sufficient features of tuples to achieve high-quality ER results. To ease the understanding of these two types of approaches, we detail their respective strengths and limitations in the following.

For the sentence-based methods, the embedding of a tuple is highly relevant to its serialized attribute values. It is resilient to anomalous values caused by data extraction errors. Take the tuple $e_{1}^{\prime}$ in Figure 1 as an example. The attribute value of manufacturer (i.e., "aspyr media inc") appears in a different place (as a part of attribute title instead of manufacturer), due to data extraction errors. The sentencebased methods treat the tuple $e_{1}^{\prime}$ as a sentence "aspyr, media, inc, sims, 2, glamour, life, stuff, pack, 23.44". In other words, "aspyr media inc" is still a part of the context of $e_{1}^{\prime}$ and can provide effective information to learn the embedding of $e_{1}^{\prime}$. Despite the benefit, two main limitations exist in the sentence-based methods. First, recent work [5] clarifies that, tuples are not sentences, and hence, treating a tuple blindly as a sentence loses a large amount of contextual information present in the tuple. Second, they dismiss the rich set of semantics inherent among different tuples [5]. To be more specific, they assume that different tuples are mutually independent. On the contrary, it is common that different tuples share the same attribute values, and some common attribute values might appear in many tuples. As shown in Figure 1, the attribute value "aspyr media" exists in both tuple $e_{1}$ and tuple $e_{2}$.

On the other hand, the graph-based ER approaches bring two benefits. First, it can capture the semantic relationships
between different attributes within every tuple. Second, it can discover the rich set of semantics inherent among different tuples. Recent studies [5], [23] transform every dataset containing a collection of tuples into a graph composed of three types of nodes, i.e., tuple-level nodes, attribute-level nodes, and value-level nodes. The graph exhibits two characteristics: (i) there is an edge between a tuple-level node and a value-level node as long as the value appears in the tuple; and (ii) there is an edge between an attribute-level node and a value-level node if the value belongs to the domain of this attribute. However, graph-based ER is error-prone. Take the sampled Amazon dataset in Figure 1 as an example. The value "aspyr media inc", which corresponds to a wrong attribute-level node, will result in a wrong graph structure. The wrong graph features might be propagated along the edges and nodes, and thus lead to unreliable embeddings of tuples. Consequently, we are required to find sufficient tuple features in order to equip graph-based ER approaches with fault-tolerance.
Contributions. The obstruction with the existing ER methods inspires us to ask a question: would it be possible to perform ER in a self-supervised manner, where reliable labels are automatically generated and sufficient entities features are captured, so that the above two challenges could be well addressed? Accordingly, we propose CollaborER, a self-supervised entity resolution, powered by multi-features collaboration. CollaborER features a sequential modular architecture consisting of two phases, i.e., automatic label generation (ALG) and collaborative ER training (CERT). In the first phase, ALG is developed to generate reliable ER labels on every dataset automatically. In the second phase, with the guidance of the generated labels, CERT learns the matching signals by utilizing both sentence features and graph features of tuples collaboratively.

We summarize the contributions of this paper as follows:

- Self-supervised ER framework. We propose a selfsupervised ER framework CollaborER, which requires zero human involvement to generate labeled tuple pairs with high quality. Once the reliable labels are generated, CollaborER produces outstanding ER results via the collaboration of both sentence features and graph features of tuples.
- Automatic label generation. We present ALG, for the first time, to automatically generate both positive and negative tuple pairs for the ER task. ALG greatly helps CollaborER to correctly identify "challenging" tuple pairs that are hard to tell whether they are matched.
- Collaborative ER training. We introduce CERT, a collaborative ER training approach, to discover both graph features and sentence features to learn sufficient tuple features for ER without sacrificing the faulttolerance capability in handling noisy tuples.
- Extensive experiments. Comprehensive evaluation over eight existing ER benchmarks demonstrates the superiority of CollaborER. It outperforms all the existing unsupervised methods. Furthermore, it is comparable with or even superior to DITTO [25], the state-of-the-art supervised ER method.
Organization. The rest of the paper is organized as follows.

TABLE 1
Symbols and description

| Notation | Description |
| :--: | :--: |
| $T$ | a relational dataset |
| $e \in T$ | a tuple belonging to the dataset $T$ |
| $A$ | a set of attribute values |
| $e . A[m]$ | the $m$-th attribute value of tuple $e$ |
| $\mathcal{G}$ | a multi-relational graph |
| $N$ | a set of nodes belonging to the graph $\mathcal{G}$ |
| $E$ | a set of edges belonging to the graph $\mathcal{G}$ |

Section 2 covers the basic background techniques used in the paper. Section 3 presents the overall architecture of our proposed CollaborER, and Section 4 and Section 5 detail the two key phases of CollaborER respectively. Section 6 reports the experimental results and our findings. Section 7 presents an anomaly detection demonstration based on the proposed CollaborER. Section 8 reviews the related work. Finally, Section 9 concludes the paper.

## 2 PRELIMINARIES

In this section, we first formalize the problem of entity resolution and then overview some background materials and techniques to be used in subsequent sections. Table 1 summarizes the symbols that are frequently used throughout this paper.

### 2.1 Problem Definition

Let $T$ be a relational dataset with $|T|$ tuples and $m$ attributes $A=\{A[1], A[2], \cdots, A[m]\}$. Each tuple $e \in T$ consists of a set of attribute values, denoted as $V=$ $\{e . A[1], e . A[2], \cdots, e . A[m]\}$. Here, $e . A[m]$ is the $m$-th attribute value of tuple $e$, corresponding to attribute $A[m] \in$ $A$. Entity resolution (ER), also known as entity matching or record linkage, aims to find matched tuple pairs $\mathcal{M}$ that refer to the same real-world entity between two relational datasets $T$ and $T^{\prime}$. The ER task can be formulated as $\mathcal{M}=\left\{\left(e, e^{\prime}\right) \in T \times T^{\prime} \mid e \equiv e^{\prime}\right\}$, where $e \in T, e^{\prime} \in T^{\prime}$, and $\equiv$ represents a matched relationship between tuples $e$ and $e^{\prime}$.

To reduce the quadratic number of candidates of matched tuple pairs, an ER program often executes a blocking phase followed by a matching phase. The goal of blocking is to identify a small subset of $T \times T^{\prime}$ for candidate pairs of high probability to be matched. In addition, blocking mechanisms are expected to have zero false negative, a common assumption made by many ER techniques [25], [33], [46]. Designing an effective blocking strategy is orthogonal to CollaborER, and we apply a common blocking method that is widely used in the existing ER approaches [20], [25], [32]. The goal of matching is to predict the matched tuple pairs in the candidate pairs, which is the focus of this work.

### 2.2 Pre-trained Language Models

Pre-trained language models (LMs), such as BERT [8] and XLNet [47], have demonstrated a powerful semantic expression ability. Based on pre-trained LMs, we can support many downstream tasks (e.g., classification and question answering). Concretely, we can plug appropriate inputs and
outputs into a pre-trained LM based on the specific task and then fine-tune all the model's parameters end-to-end.

Intuitively, the ER problem can be treated as a sentence pair classification task [25]. Given two tuples $e_{i} \in T$ and $e_{j}^{\prime} \in T^{\prime}$, pre-trained LMs transform them into two sentences $\mathcal{S}\left(e_{i}\right)$ and $\mathcal{S}\left(e_{j}^{\prime}\right)$, respectively. A sentence $\mathcal{S}\left(e_{i}\right)$ is denoted by $\mathcal{S}\left(e_{i}\right)::=\left\langle\left[\right.\right.$ COL] $A[1]$ [VAL] $e_{i} \cdot A[1] \ldots$ [COL] $A[m]$ [VAL] $e_{i} \cdot A[m]$ ), where [COL] and [VAL] are special tokens for indicating the start of attribute names and the start of attribute values, respectively. Note that, we exclude missing values and their corresponding attribute names from the sentence since they contain zero valid information. A tuple pair $\left(e_{i}, e_{j}^{\prime}\right)$ can be serialized as a pairwise sentence $\mathcal{S}\left(e_{i}, e_{j}^{\prime}\right)::=\left\langle\mathcal{S}\left(e_{i}\right)\right.$ [SEP] $\mathcal{S}\left(e_{j}^{\prime}\right\rangle$ ), where [SEP] is a special token separating the two sentences. For the sentence pair classification task, pre-trained LMs take each pairwise sentence $\mathcal{S}\left(e_{i}, e_{j}^{\prime}\right)$ as an input. Note that, a special symbol [CLS] is added in front of every input sentence, which is utilized to store the classification output signals during the fine-tuning of LMs.
Objective Function. We employ CrossEntropy Loss, a widely used classification objective function, to fine-tune the pretrained LMs in CollaborER. CrossEntropy Loss is designed to keep the predicted class labels similar to the ground-truth. Formally,

$$
\mathcal{L}\left(y=k \mid \mathcal{S}\left(e_{i}, e_{j}^{\prime}\right)\right)=-\log \left(\frac{\exp \left(d_{k}\right)}{\sum_{q}^{|k|} \exp \left(d_{q}\right)}\right) \forall k \in\{0,1\}
$$

Here, $\boldsymbol{d} \in \mathbb{R}^{|k|}$ is the logits computed by $\boldsymbol{d}=\mathbf{W}_{c}^{\top} \mathbf{E}_{[\mathbf{C L S}]}$. $\mathbf{W}_{c} \in \mathbb{R}^{n \times|k|}$ is a learnable linear matrix, where $n$ is the dimension of the sentence embeddings. $\mathbf{E}_{[\mathbf{C L S}]}$ is the embedding of the symbol [CLS]. For sentence pair classification, the class labels are binary $\{0,1\}$. We denote $y=1$ a truly matched pair and $y=0$ a mismatched pair.

### 2.3 Graph Neural Networks

Graph neural networks (GNNs) are popular graph-based models, which capture graph features via message passing between the nodes of graphs. GNNs are suitable for the ER task because of the following two aspects. First, GNNs ignore the sequence relationship between different attributes but discover the features of each tuple by aggregating the semantic information contained in the corresponding attribute names and values. This conforms to the real characteristics of the relational dataset since a tuple is not a sentence and entities' attributes can be organized in any order. Thus, GNNs are able to effectively capture the features within every tuple. Second, recall that GNNs capture graph features via message passing between relevant nodes, i.e., the set of tuples sharing the same attribute values in this paper. Accordingly, GNNs have the capability of learning rich semantics among those relevant tuples, since the features of a tuple can be passed to another tuple through an edge (i.e., a shared attribute value). The core idea of GNNs is to learn each node representation by capturing the information passing from its neighborhoods. Generally, GNNs learn the

![img-1.jpeg](img-1.jpeg)

Fig. 2. CollaborER framework

embeddings of each node $n_i$ obeying the following equations [28]:

$$
\mathbf{o}_{i}^{l+1} = \text{AGGREGATION}^l \left( \left[ \left( \left( \mathbf{h}_{j}^{l}, \mathbf{r}_{i,j} \right) : j \in \mathcal{N}(i) \right) \right] \right) \tag{2}
$$

$$
\mathbf{h}_{i}^{l+1} = \text{UPDATE}^{l+1} \left( \mathbf{h}_{i}^{l}, \mathbf{o}_{i}^{l+1} \right) \tag{3}
$$

where $\mathbf{h}_{i}^{l}$ represents the embedding of the $l$-th layer node $n_i$, $\mathbf{r}_{i,j}$ stands for the embedding of an edge that connects the node $n_i$ and another node $n_j$, and $\{\cdots\}$ denotes a multiset. $\mathcal{N}(i)$ represents the set of neighboring nodes around $e_i$. Eq. (2) is to aggregate information from the neighboring nodes while Eq. (3) transforms the entity embeddings into better ones. To serve the purpose of AGGREGATION, we can use graph convolutional network (GCN) [19] or graph attention network (GAT) [43].

## 3 FRAMEWORK OVERVIEW

In this section, we overview the framework of CollaborER, as illustrated in Figure 2. CollaborER consists of two phases, i.e., (i) automatic label generation (ALG), and (ii) collaborative ER training (CERT).

**Automatic label generation (ALG).** As mentioned in Section 1, pre-collected ER labels are often not available in many real-world scenarios. It inspires us to look for ways to generate approximate labels via an automatic label generation program. Given two datasets $T$ and $T'$ each of which contains a collection of tuples, this phase is to generate pseudo-labels with high-quality, including both *positive labels* and *negative labels*, for the guidance of the subsequent CERT process.

Positive labels refer to a set of positive tuple pairs, denoted as $\mathbb{P}$. For each positive tuple pair $\mathbb{P}(e_i, e'_i)$, the tuple $e_i \in T$ and the tuple $e'_i \in T'$ have a high probability of being matched. In ALG, we introduce a *reliable positive label generation (RPLG)* strategy to obtain positive tuple pairs with high confidence.

On the other hand, negative labels refer to a set of negative tuple pairs, denoted as $\mathbb{N}$. For each negative tuple pair $\mathbb{N}(e_i, e'_i)$, the tuple $e_i \in T$ and the tuple $e'_i \in T' - \{e'_i\}$ are unlikely to be matched. Random sampling [31], [42] is a widely-used approach for generating negative labels. Given a positive tuple pair $\mathbb{P}(e_i, e'_i)$, random sampling replaces either $e_i$ or $e'_i$ with an arbitrary tuple. However, recent studies [39], [40] have indicated that the randomly generated negative tuple pairs are easily distinguished from positive ones. For instance, if we generate a negative tuple pair ("Apple Inc.", "Google") for a positive tuple pair ("Apple Inc.", "Apple"), it is obvious that "Google" and "Apple Inc." are not equivalent. These negative tuple pairs are uninformative, and contribute little to the embedding training process. Ideally, an effective negative label generation is expected to put two similar tuples (but they are not related to the same real-world entity) into a pair. It facilitates an ER-oriented embedding model (e.g., CERT in this paper) to be capable of identifying whether two entities of a "challenging" tuple pair refer to the same real-world entity. To this end, we propose a *similarity-based negative label generation (SNLG)* method in ALG to generate negative labels with semantic similarity.

**Collaborative ER training (CERT).** Recall that matching entities purely based on the *sentence features* or the *graph features* of tuples results in insufficient feature discovery or erroneous feature involvement. The goal of CERT is to capture and integrate both the sentence features and the graph features of tuples in a unified framework to improve the quality of ER results.

Given two datasets $T$ and $T'$ and a set of labels (including positive labels $\mathbb{P}$ and negative labels $\mathbb{N}$) generated by ALG, CERT first introduces *multi-relational graph construction (MRGC)* to construct a multi-relational graph $\mathcal{G}$ (w.r.t. $\mathcal{G'}$) for each dataset $T$ (w.r.t. $T'$). We would like to highlight that the graph structure generated by the proposed MRGC is much simpler than that generated by other existing ER methods (e.g., EMBDI [5] and GraphER [23]) without losing the expressive power of tuples' graph features, as confirmed in the experimental evaluations to be presented in Section 6.5.1.

Then, CERT learns the embeddings of each tuple based on the graph structure. CERT is treated as a black box, such that users could enjoy the flexibility of applying their choice of graph-based models to embed both nodes and edges in a multi-relational graph. Our current implementation utilizes AttrGNN [27] for this purpose. Afterward, we feed the well-trained graph features (i.e., embeddings) of tuples into a pre-trained language model (LM) to assist the learning of the sentence features of tuples. More specifically, the graph features of tuples are used to complement the semantic features of tuples that cannot be captured by a sentence-based model.

## 4 AUTOMATIC LABEL GENERATION (ALG)

In this section, we present an automatic label generation (ALG) strategy. It contains two components, including (i) a reliable positive label generation (RPLG) method and (ii) a similarity-based negative label generation (SNLG) method.

Generating either positive labels or negative labels is highly relevant to the similarity between tuples. Motivated by the powerful capability of semantics expression of pre-trained language models, we leverage sentence-BERT [35], a variant of BERT that achieves outstanding performance for semantic similarity search, to assign a pre-trained embedding for each tuple. In general, different similarity functions (e.g., cosine distance and Manhattan distance) can be

applied to quantify the semantic similarity between tuples from different datasets in ALG, according to the characteristics of the datasets. In the current implementation, we find empirically that cosine distance brings considerable performance. To this end, we choose it as the similarity function in ALG. The tuple similarity matrix is denoted as $\mathbf{M}_{t} \in\left[0,1|^{\left|T\right| \times\left|T^{\prime}\right|}\right.$, where $|T|$ and $\left|T^{\prime}\right|$ represent the total number of tuples in $T$ and $T^{\prime}$ respectively. In the following, we detail how to generate positive and negative labels via RPLG and SNLG, respectively.

### 4.1 Reliable Positive Label Generation (RPLG)

RPLG aims to find positive tuple pairs with a high probability of being matched. A common approach is to consider tuples that are mutually most similar to each other. However, we find empirically many mutually similar tuples do not refer to the same entities. Considering that high-quality labels are essential for embedding-based ER model as wrong labels will mislead the ER model training, we choose to generate the positive labels by IKGC [48], which gives much stronger constraints to ensure the high-quality of positive labels than the methods based on the mutual similarity. It generates tuple pairs as positive labels that satisfy two requirements [48], including (i) they are mutually most similar to each other; and (ii) there is a margin between, for each tuple $e$, its most similar tuple and the second most similar one.

Specifically, for each tuple $e_{i} \in T$, we assume that $e_{j}^{\prime}, e_{k}^{\prime} \in T^{\prime}$ are the most similar and the second most similar tuples in $T^{\prime}$ to $e_{i}$, respectively. Similarly, for tuple $e_{j}^{\prime} \in T^{\prime}$ (i.e., the most similar tuple to $e_{i} \in T$ ), let $e_{l}$ and $e_{u}$ denote its most similar tuple in $T$ and the second most similar tuple in $T$ respectively. If tuple pair $\left(e_{i}, e_{j}^{\prime}\right)$ could be considered as a positive label, we expect $e_{i}=e_{l}$, i.e., $e_{i}$ and $e_{j}^{\prime}$ are mutually most similar to each, i.e., the requirement (i) stated above. In addition, their similarity discrepancies $\delta_{1}=$ $\operatorname{Sim}\left(e_{i}, e_{j}^{\prime}\right)-\operatorname{Sim}\left(e_{i}, e_{k}^{\prime}\right)$ and $\delta_{2}=\operatorname{Sim}\left(e_{j}^{\prime}, e_{l}\right)-\operatorname{Sim}\left(e_{j}^{\prime}, e_{u}\right)$ are expected to be both above a given threshold $\theta$, i.e., the requirement (ii) stated above. Here, $\operatorname{Sim}\left(e, e^{\prime}\right)$ denotes the similarity score between two tuples $e \in T$ and $e^{\prime} \in T^{\prime}$.
Discussion. We would like to emphasize that RPLG is a general approach, which can be easily integrated into various ER methods. RPLG is able to not only generate positive labels with high-quality (see the experiments to be presented in Section 6.5.2) but also achieve desirable ER results without any time-consuming training process (see the experiments to be presented in Section 6.4).

### 4.2 Similarity-based Negative Label Generation (SNLG)

As the random-based negative label generation method has rather limited contribution to the embedding-based ER training, it is essential to generate more "challenging" negative labels, as described in Section 3. To achieve this goal, we propose a similarity-based negative label generation (SNLG) strategy. Given a positive tuple pair $\mathbb{P}\left(e_{i}, e_{i}^{\prime}\right)$, where $e_{i} \in T$ and $e_{i}^{\prime} \in T^{\prime}$, SNLG generates a set of negative labels $\mathbb{N}\left(e_{i}, e_{i}^{\prime}\right)$ by replacing either $e_{i}$ or $e_{i}^{\prime}$ with its $\epsilon$-nearest neighborhood in the semantic embedding space. Again, we use the cosine similarity metric to search for the $\epsilon$-nearest neighbors of $e_{i}$ and $e_{i}^{\prime}$, respectively.

```
Algorithm 1: Multi-Relational Graph Construction
    Input: a relational dataset \(T\)
    Output: a multi-relational graph \(\mathcal{G}\)
    \(\mathcal{G} \longleftarrow \varnothing\)
    foreach \(e_{i} \in T\) do
        \(\mathcal{G} \cdot \operatorname{addNode}\left(e_{i}\right)\)
        foreach \(v_{j} \in\left\{e_{i} \cdot A[1], e_{i} \cdot A[2], \cdots, e_{i} \cdot A[m]\right\}\) do
            if \(v_{j}\) is not included in \(\mathcal{G}\) then
                \(\mathcal{G} \cdot \operatorname{addNode}\left(v_{j}\right)\)
            \(a_{i, j} \longleftarrow\) find the attribute name of \(v_{j}\)
            \(\mathcal{G} \cdot \operatorname{addEdge}\left(e_{i}, a_{i, j}, v_{j}\right)\)
```

9 return $\mathcal{G}$

Discussion. Even though this is a very intuitive and simple method, it effectively promotes the performance of Colla$\operatorname{borER}$. We will demonstrate the superiority of using the proposed SNLG to generate negative labels for ER in the experiments to be presented in Section 6.4.

## 5 Collaborative ER Training (CERT)

This section details a newly proposed collaborative ER training (CERT) approach to discover the features of tuples from both the graph aspect and the sentence aspect to facilitate the ER process. CERT is composed of two phases, i.e., (i) multi-relational graph feature learning (MRGFL) and (ii) collaborative sentence feature learning (CSFL).

### 5.1 Multi-Relational Graph Feature Learning (MRGFL)

Inspired by the graph structure's powerful capturing ability of semantics, we propose a multi-relational graph feature learning method (MRGFL) to represent tuples according to their graph features. It first proposes a multi-relational graph construction (MRGC) approach for transforming datasets from the relational format to the graph structure, and it then learns the tuple representations via a GNN-based model, e.g., AttrGNN [27] in our current implementation.
Multi-relational graph construction (MRGC). Graph construction techniques have been presented in the existing ER work, such as EMBDI [5] and GraphER [23]. These techniques treat tuples, attribute values, and attribute names as three different types of nodes. Edges exist if there are relationships between nodes. Nonetheless, several drawbacks restrict the scope of using these graph construction methods to perform ER in real-world scenarios.

First, these graph construction approaches may produce intricately large-scale graphs containing a large number of edges and nodes. Storing a graph with massive edges and nodes is memory-consuming, and meanwhile training a graph embedding model (e.g., GNN) on a large graph is challenging too, as widely-acknowledged by many existing works [6], [16], [52].

Second, these graph construction methods lack consideration of the semantics contained in an edge itself. For instance, assume that there are two types of edges, i.e., attribute-value edges and tuple-attribute edges. The former represents an edge that connects an attribute-level node with a

![img-2.jpeg](img-2.jpeg)

Fig. 3. A motivating example of proposing the multi-relational graph construction (MRGC)

value-level node; the latter represents an edge connecting a tuple-level node with an attribute-level node. It is intuitive that these two types of edges have different semantic meanings, and hence, they should be considered differently when learning features of tuples in the ER task.

The above limitations motivate us to design a relatively **small-scale** but highly **effective** MRGC to construct a multi-relational graph for every dataset. We start by defining a multi-relational graph, formally $$G = \{N, E, A\}$$. Here, *N* and *E* refer to a set of nodes and a set of edges, respectively; and *A* represents the set of attributes corresponding to the nodes and the edges. There are two types of nodes in *G*, i.e., *tuple-level nodes* and *value-level nodes*. A tuple-level node represents a tuple *e*; while a value-level node corresponds to an attribute value *v* in a relational dataset. Each attribute *a* ∈ *A* denotes an attribute name in the relational dataset. *E* = {(*e*, *a*, *v*)|*e*, *v* ∈ *N*, *a* ∈ *A*} represents the set of edges. Each edge connects a tuple-level node *e* with a value-level node *v* via an attribute *a*, meaning that *e* has *v* as its value for attribute *a*.

Next, we describe the MRGC procedure, with its pseudo code presented in Algorithm 1. Given a relational dataset *T*, MRGC initializes an empty multi-relational graph *G* (Line 1). Then, MRGC iteratively adds nodes and edges to *G* (Lines 2-8). Specifically, for every tuple *e<sub>i</sub>* ∈ *T*, MRGC first selects its tuple Id as a tuple-level node (Line 3) and then adds a set of value-level nodes that correspond to this tuple (Lines 4-6). Note that, since different tuples share the same attribute names, MRGC generates a set of edges for *e<sub>i</sub>*, with each connecting the tuple-level node of *e<sub>i</sub>* and a value-level node *v<sub>j</sub>* ∈ {*e<sub>i</sub>*.*A*[1], *e<sub>i</sub>*.*A*[2], ..., *e<sub>i</sub>*.*A*[*m*]}, denoted as (*e<sub>i</sub>*, *a<sub>i,j</sub>*, *v<sub>j</sub>*).

**Discussion.** Compared to the existing graph construction methods, MRGC constructs a small graph that is still able to well preserve the semantics of tuples. Take the sampled Amazon dataset as an example. Figure 3 shows the respective graph structures constructed by three different graph construction methods, including EMBDI [5], GraphER [23], and the proposed MRGC in this paper. It is obvious that the graph constructed by MRGC is the smallest, containing fewer nodes and edges than other graphs. Also, we will verify the small-scale characteristics of MRGC in the experiments to be presented in Section 6.5.1. Besides, MRGC not only preserves the semantic relationships between each tuple and its corresponding attribute values, but also maintains semantic connections between different tuples by connecting them with a shared value-level node. For example, *e<sub>1</sub>* and *e<sub>2</sub>* have semantic connections since they both have edges linking to the same value-level node "aspyr media".

**Multi-relational graph feature learning (MRGFL).** Given two multi-relational graphs *G* (w.r.t. *T*) and *G*′ (w.r.t. *T′*), MRGFL aims to embed tuples from different sources in a unified vector space by considering their graph structures. In this vector space, matched tuples are expected to be as close to each other as possible. Generally, we can treat MRGFL as a graph-based ER problem, which is highly relevant to *entity alignment* (EA) [41] that aims to find a correspondence between entities from different multi-relational graphs. To this end, MRGFL is regarded as a black box. Users have the flexibility to learn the embeddings of tuples by applying any available EA model [24], [27], [41]. In our implementation, we adopt the state-of-the-art EA model AttrGNN [27] that aggregates the graph feature of each tuple via multiple newly proposed GNNs, for this purpose. In the following, we sketch the main idea about how to use an EA model to learn tuples' graph features in MRGFL.

First, the graph features of each tuple can be obtained by applying a GNN model, as described in Section 2.3. It outputs a set of tuples' embeddings. We denote the embedding of each tuple *e<sub>i</sub>* ∈ *T* (w.r.t. *e*′<sub>i</sub> ∈ *T′*) as **h<sub>e<sub>i</sub></sub>** (w.r.t. **h<sub>e</sub><sup>c</sup>**). Then, a *training objective function* (denoted as *L<sub>g</sub>*) is used to unify the two datasets' tuple embeddings into a unified vector space by maximizing the similarities with regard to the generated labels. Formally,

$$L_g = \sum_{(e_i, e'_i) \in \mathbb{P}} \sum_{(e_j, e_k) \in \mathbb{N}} [d(e_i, e'_i) + \gamma - d(e_j, e_k)]_{+}\tag{4}$$

Here, (*e<sub>i</sub>*, *e*′<sub>i</sub>) ∈ **P** represents a positive label; (*e<sub>j</sub>*, *e<sub>k</sub>*) ∈ **N** represents a negative label; [*b*]<sub>+</sub> = max{0, *b*}; *d*(*e<sub>i</sub>*, *e*′<sub>i</sub>) denotes the cosine distance between **h<sub>e<sub>i</sub></sub>** and **h<sub>e</sub><sup>c</sup>**, where **h<sub>e<sub>i</sub></sub>** and **h<sub>e</sub><sup>c</sup>** are the final embeddings of *e<sub>i</sub>* and *e*′<sub>i</sub> w.r.t. the multi-relational graph *G* after performing the |*l*|-th layer GNN model, respectively; similarly, *d*(*e<sub>j</sub>*, *e<sub>k</sub>*) represents the cosine distance between **h<sub>e<sub>i</sub></sub>** and **h<sub>e</sub><sup>k</sup>**; and γ is a margin hyperparameter. We set γ = 1.0 in the current implementation.

![img-3.jpeg](img-3.jpeg)

Fig. 4. The architecture of CSFL

### 5.2 Collaborative Sentence Feature Learning (CSFL)

Recent studies [10], [25], [51] have demonstrated that capturing the sentence features of tuples can help ER task to a certain degree. Nonetheless, treating tuples as sentences causes insufficient feature discovery, as mentioned in Section 1. In view of this, we propose a collaborative sentence feature learning (CSFL) model, which discovers sufficient tuples' sentence features for ER with the assistance of the well-trained graph features of tuples. The training objective of CSFL is to (i) identify whether two tuples refer to the same real-world entity; and (ii) minimize the semantic distance between the matched tuples. The architecture of CSFL is depicted in Figure 4.

First, we present how to identify the matched (or mismatched) tuples in CSFL. We fine-tune a pre-trained LM with a sentence pair classification task. We take as inputs a pairwise sentence $\mathcal{S}\left(e_{i}, e_{i}^{\prime}\right)$ and its corresponding positive and negative labels generated by the proposed ALG. Then, we learn the classification signal $\mathbf{E}_{[\mathbf{C L S}]}$ by feeding the inputs into a multi-layer Transformer encoder. In the current implementation, the number of transformer layers is set to 12, a typical setting used in various tasks. We use a variant of CrossEntropy Loss $\mathcal{L}_{1}$ as the objective training function, which is derived from Equation (1). Formally,

$$
\begin{gathered}
\mathcal{L}_{1}\left(y=k \mid \mathcal{S}\left(e_{i}, e_{j}^{\prime}\right)\right)=-\log \left(\frac{\exp \left(d_{k}^{*}\right)}{\sum_{q}^{|k|} \exp \left(d_{q}^{*}\right)}\right) \forall k \in\{0,1\} \\
\boldsymbol{d}^{*}=\mathbf{W}_{c}^{s \top}\left(\mathbf{E}_{[\mathbf{C L S}]} ; \mathbf{G}_{\mathbf{a b s}} ; \mathbf{G}_{\mathbf{d o t}}\right)
\end{gathered}
$$

Here, the logits $\boldsymbol{d}^{*}$ is produced by both tuples' sentence features (i.e., $\mathbf{E}_{[\mathbf{C L S}]}$ ) and tuples' graph features (i.e., $\mathbf{G}_{\mathbf{a b s}} \in$ $\mathbb{R}^{c}$ and $\mathbf{G}_{\mathbf{d o t}} \in \mathbb{R}^{c}$ ), where $c$ is the dimension of the tuples' graph features. $\mathbf{W}_{c}^{*} \in \mathbb{R}^{(n+2 c) \times|k|}, \mathbf{G}_{\mathbf{a b s}}=\left|\mathbf{h}_{e_{i}}-\mathbf{h}_{e_{i}^{\prime}}\right|$ denotes the element-wise difference between the graphbased embeddings $\mathbf{h}_{e_{i}}$ and $\mathbf{h}_{e_{i}^{\prime}} . \mathbf{G}_{\mathbf{d o t}}=\mathbf{h}_{e_{i}} \otimes \mathbf{h}_{e_{i}^{\prime}}$ represents the element-wise similarity between $\mathbf{h}_{e_{i}}$ and $\mathbf{h}_{e_{i}^{\prime}}$.

Second, we illustrate how to minimize the semantic distance between the matched tuples. At the input, the pre-trained LM allocates an initialized embedding for each token of a sentence $\mathcal{S}\left(e_{i}\right)$ (w.r.t. $\mathcal{S}\left(e_{i}^{\prime}\right)$ ), denoted as $\mathbf{E}_{\mathbf{i}}$ (w.r.t. $\mathbf{E}_{\mathbf{i}}^{\prime}$ ). Note that, the special symbol [CLS], which is located in the front of every sentence, also has an initial embedding, denoted as $\mathbf{E}_{[\mathbf{C L S}]}$. The embedding of every token will be updated after performing the multi-layer Transformer encoder. We apply max-pooling to obtain a fixedlength embedding $\mathbf{E}_{\mathbf{e}_{\mathbf{i}}}$ (w.r.t. $\mathbf{E}_{\mathbf{e}_{\mathbf{i}}^{\prime}}$ ) for representing the tuple $e$ (w.r.t. $e^{\prime}$ ). Concretely, max-pooling generates the fixedlength embedding by selecting the maximal value in each dimension among all the embedded tokens of the tuple. We use CosineEmbedding Loss $\mathcal{L}_{2}$ as the objective training function. It is designed to minimize the semantic distance between matched tuples (w.r.t. the set of positive labels $\mathbb{P}$ ) and maximize that between mismatched tuples (w.r.t. the set of negative labels $\mathbb{N}$ ). Formally,

$$
\mathcal{L}_{2}\left(y \mid \mathcal{S}\left(e_{i}\right), \mathcal{S}\left(e_{j}^{\prime}\right)\right)=\left\{\begin{array}{l}
1-\cos \left(\mathbf{E}_{\mathbf{e}_{\mathbf{i}}}, \mathbf{E}_{\mathbf{e}_{\mathbf{j}}^{\prime}}\right), \text { if } y=1 \\
\max \left(0, \cos \left(\mathbf{E}_{\mathbf{e}_{\mathbf{i}}}, \mathbf{E}_{\mathbf{e}_{\mathbf{j}}^{\prime}}\right)-\lambda\right), \text { if } y=0
\end{array}\right.
$$

where $\lambda$ is a margin hyper-parameter separating matched tuple pairs from mismatched tuple pairs. $\cos (\cdot, \cdot)$ represents the cosine distance metric.

Finally, we are ready to present the overall training function of CSFL, namely, CollaborER training loss (denoted as $\mathcal{L}_{c}$ ). Formally,

$$
\mathcal{L}_{c}=\mathcal{L}_{1}+\mu \mathcal{L}_{2}
$$

where hyper-parameter $\mu \in[0,1]$ is a coefficient controlling the relative weight of $\mathcal{L}_{2}$ against $\mathcal{L}_{1}$. The ER results can be obtained according to the predicted labels of each tuple pair.
Discussion. Compared to the existing sentence-based ER methods that also fine-tune pre-trained LMs, we emphasize the superiority of CSFL in the following two aspects. First, CSFL incorporates the graph features of tuples learned in the previous MRGFL step to enrich the features that the sentence-based model fails to capture. Second, we argue that utilizing the CosineEmbedding Loss (i.e., $\mathcal{L}_{2}$ defined in Equation 7) as a part of CollaborER training loss is suitable for the ER task. Intuitively, matched tuples should have similar embeddings in a unified semantic vector space. However, the existing sentence-based ER methods, which fine-tune and cast ER as a sentence-pair classification problem, cannot ensure the semantic similarity between matched tuples. We will verify the superiority of the proposed CSFL in the experiments to be presented in Section 6.4.

## 6 EXPERIMENTS

In this section, we conduct comprehensive experiments to verify the performance of CollaborER from the following three aspects. First, we compare CollaborER with several competing ER approaches and present the results in Section 6.3. Second, we conduct the ablation study for the proposed CollaborER and report our findings in Section 6.4. Third, we further explore CollaborER by (i) comparing the scale of the graphs generated by the proposed multirelational graph construction (MRGC) method and other existing approaches in Section 6.5.1; and (ii) analyzing the performance of both the reliable positive label generation (RPLG) and the similarity-based negative label generation (SNLG) (in the automatic label generation (ALG) strategy) in Section 6.5.2.

TABLE 2
Statistics of the datasets used in experiments

| Type | Dataset | Domain | \#Attr. | \#Domain | \#Tuple | \#Pos. |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Structured | AG | software | 3 | $123-3,021$ | $1,363-3,326$ | 1,167 |
|  | BR | beer | 4 | $4-4,343$ | $4,345-3,000$ | 68 |
|  | DA-clean | citation | 4 | $5-2,507$ | 2,616-2,294 | 2,220 |
|  | FZ | restaurant | 6 | $16-533$ | $533-331$ | 110 |
|  | IA-clean | music | 8 | $6-38,794$ | $6,907-55,923$ | 132 |
| Dirty | DA-dirty | citation | 4 | $6-2,588$ | 2,616-2,294 | 2,220 |
|  | IA-dirty | music | 8 | $7-55,727$ | $6,907-55,923$ | 132 |
| Textual | AB | product | 3 | $167-1,081$ | $1,081-1,092$ | 1,028 |

### 6.1 Benchmark Datasets

We conduct experiments on eight representative and widelyused ER benchmarks with different sizes and in various domains. Table 2 lists the detailed statistics. For structured ER, we use five benchmarks, including Amazon-Google (AG), BeerAdvo-RateBeer (BR), the clean version of DBLPACM (DA-clean), Fodors-Zagats (FZ), and the clean version of iTunes-Amazon (IA-clean). The attribute values of tuples are atomic but not a composition of multiple values. For dirty ER, following [32], we use the dirty versions of the DBLP-ACM and iTunes-Amazon benchmarks to measure the robustness of the proposed CollaborER against noise. For textual ER, we use the Abt-Buy (AB) benchmark which is text-heavy, meaning that at least one attribute of each tuple contains long textual values.

### 6.2 Implementation and Experimental setup

Evaluation metric. To measure the quality of ER results, we use F1-score, the harmonic mean of precision (Prec.) and recall (Rec.) computed as $\frac{2 \times(\text { Prec. } \times \text { Rec. })}{(\text { Prec. }+ \text { Rec. })}$. Here, precision is defined as the fraction of match predictions that are correct; and recall is defined as the fraction of real matches being predicted as matches.
Competitors. We compare CollaborER against 6 SOTA ER approaches. The competitors can be classified into two categories based on whether pre-defined lables are required, i.e., unsupervised ER and supervised ER.

The former refers to the group of approaches that performs ER without any label involvement, including (i) ZeroER [46], a powerful generative ER approach based on Gaussian Mixture Models for learning the match and unmatch distributions; and (ii) EMBDI [5], which automatically learns local embeddings of tuples for ER based on the attribute-centric graphs. Methods in this group are most relevant to CollaborER.

The latter refers to the group of approaches that relies on the pre-defined labels for matching tuples, including (i) DeepMatcher+ (DM+) [25], which implements multiple ER methods and reports the best performance (highest F1-scores), including DeepER [10], Magellan [20], DeepMatcher [32], and DeepMatcher's follow-up work [13] and [18]; (ii) GraphER [23], which integrates schematic and structural information into token representations with a GNN model for ER and aggregates token-level features as the ER results; (iii) MCA [50], which incorporates attention mechanism into a sequence-based model to learn features of tuples for ER; (iv) ERGAN [36], which employs a generative adversarial network to augment labels and predict whether
two entities are matched; and (v) DITTO [25], which leverages a pre-trained Transformer-based language model to fine-tune and cast ER as a sentence-pair classification problem. Approaches in this group are used to demonstrate that the proposed CollaborER, although not requiring any laborintensive annotations/labels, is able to achieve performance that is comparable with or even better than the performance achieved by SOTA supervised ER in various real-world ER scenarios.

Note that, in the evaluation of supervised ER methods, each dataset is split into the training, validation, and test sets using the ratio of 3:1:1. For fair comparisons with supervised methods, we report the results conducted by CollaborER on the test sets, denoted as CollaborER-S. For fair comparisons with unsupervised methods, we report the results of CollaborER on the whole datasets, denoted as CollaborER-U.
Implementation details. We implemented CollaborER ${ }^{1}$ in PyTorch [34], the Transformers library [45], and the Sentence-Transformers library [35]. In automatic label generation (ALG), we use stsb-roberta-base ${ }^{2}$ as the pre-trained LM to get the embedding for every tuple. We set $\theta=0.03$ in the process of reliable positive label generation (RPLG) and $\epsilon=10^{3}$ in the process of similarity-based negative label generation (SNLG). In collaborative ER training (CERT), the dimension of the graph feature in the process of multirelational graph feature learning (MRGFL) is 128. Besides, in both the training and test process of CollaborER, we apply the half-precision floating-point (fp16) optimization to save the GPU memory usage and the running time. In all experiments, the max sequence length is set to 256; the learning rate is set to $2 \mathrm{e}-5$; the batch size for the AG benchmark is set to 64 while that for the other benchmarks is set to 32 . The training process runs a fixed number of epochs ( $1,2,3,6$, or 30 depending on the dataset size), and returns the checkpoint at the last epoch. We set $\lambda=0.5$ and $\mu=0.2$ in the proposed CollaborER training loss. All the experiments were conducted on a personal computer with an Intel Core i9-10900K CPU, an NVIDIA GeForce RTX3090 GPU, and 128GB memory. The programs were all implemented in Python.

### 6.3 Overall Performance

Table 3 summarizes the overall ER performance of CollaborER and its competitors.
CollaborER vs. unsupervised methods. It is observed that CollaborER significantly outperforms all the unsupervised competitors. Particularly, CollaborER brings about $1 \%-85 \%$ absolute improvement over the best baseline (i.e., EMBDI). The results also demonstrate that CollaborER is more robust against data noise than ZeroER. On the dirty datasets, the performance degradation of CollaborER is only $0.67 \%$ on average. Nevertheless, the performance of ZeroER decreases by $33 \%$. The reason is that unsupervised methods can easily be fooled without the guidance of any supervision signal,

[^0]
[^0]:    1. The source code of CollaborER is available at https://github.com/ZJU-DAILY/CollaborER
    2. https://github.com/UKPLab/sentence-transformers
    3. To avoid false negative labels, we dismiss the top-2 neighbors into consideration.

TABLE 3
Overall ER results with and without any pre-defined labels (F1-score values are in percentage, and the best scores are in bold)

| Type | Datasets | Unsupervised |  | Self-supervised | Supervised |  |  |  |  | Self-supervised |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  |  | ZeroER ${ }^{1}$ | EMBDI | CollaborER-U | DM+ | GraphER | MCA | ERGAN | DITTO | DITTO-S | CollaborER-S |
| Structured | AG | 48.00 | 59.00 | 68.61 | 70.70 | 68.08 | 71.40 | 37.49 | 75.58 | 65.01 | 71.91 |
|  | BR | 51.50 | 86.00 | 87.69 | 78.80 | 79.71 | 80.00 | 74.42 | 94.37 | 90.32 | 96.55 |
|  | DA-clean | 96.00 | 95.00 | 98.63 | 98.45 | - | 98.90 | 98.51 | 98.99 | 98.33 | 98.65 |
|  | FZ | 100 | 99.00 | 100 | 100 | - | - | 98.48 | 100 | 100 | 100 |
|  | IA-clean | 82.40 | 11.00 | 96.12 | 91.20 | - | - | 77.29 | 97.06 | 94.12 | 100 |
| Dirty | DA-dirty | 63.00 | - | 98.25 | 98.10 | - | 98.50 | 81.79 | 99.03 | 98.87 | 99.10 |
|  | IA-dirty | / | - | 95.17 | 79.40 | - | - | 67.11 | 95.65 | 83.02 | 98.18 |
| Textual | AB | 52.00 | 82.50 | 83.17 | 62.80 | - | 70.80 | 30.37 | 89.33 | 78.49 | 85.01 |

${ }^{1}$ The symbol "*" represents the corresponding ER results obtained by our re-implementation with the publicly available source code.
${ }^{2}$ The symbol "/" indicates that the ER model fails to produce any result after running for 5 days in the experimental conditions.
${ }^{3}$ The symbol "-" denotes that the results are not provided in the original paper.

![img-4.jpeg](img-4.jpeg)

Fig. 5. Ablation study of CollaborER. Blue bars (corresponding to test) represent the results on the test sets, and red bars (corresponding to all) denote the results on the whole datasets.
as discussed in Section 1. On the contrary, CollaborER generates reliable labels via the proposed ALG strategy as the supervision signals. The reliability analysis of ER labels generated by ALG can be found in Section 6.5.2. Besides, the collaborative ER training process (i.e., CERT), which absorbs both graph features and sentence features of tuples, has the fault-tolerance capability for dealing with noisy tuples. The outstanding ER performance and the robust property make CollaborER more attractive in practical ER scenarios.
CollaborER vs. supervised methods. As we can see, the performance of CollaborER is comparable with or even superior to the SOTA supervised ER approaches. Concretely, CollaborER outperforms even the best supervised competitor (i.e., DITTO) by $1.54 \%$ on average over 5 datasets. Although the performance of CollaborER in the other three datasets is inferior to that of DITTO, the difference in their respective F1-scores does not exceed $4 \%$. This is really impressive since CollaborER requires zero human involvement in annotating labels for ER. In contrast, DITTO requires a sufficient amount of labels that are expensive to obtain and often times infeasible. To further compare the performance difference between CollaborER and DITTO under a fair comparison, we evaluate the performance of DITTO when using the pseudo labels generated by ALG, denoted as

DITTO-S. As can be observed, the performance of DITTOS is inferior to that of CollaborER. The reason is that, DITTO-S treats entities as sentences, and hence, it does not consider the rich semantic features of entities, as mentioned in Section 1. Nonetheless, CollaborER has the capability to capture rich semantics of entities by discovering both sentence features and graph features of entities collaboratively.

In addition, both CollaborER and ERGAN generate pseudo labels for the purpose of improving their ER performance. However, we can observe from the results that Colla$\operatorname{borER}$ achieves superior performance than ERGAN. Specifically, CollaborER brings about $23 \%$ improvement on average on the F1-score, compared to the ER results produced by ERGAN. The inferior performance of ERGAN is attributed to the inherent GAN. To be more specific, it is common that the training process of GAN is unstable [30], incurring the poor quality of the generated pseudo labels and unsatisfied training performance. By analyzing the evaluated datasets, ERGAN generates positive and negative labels with an average accuracy of $88 \%$ and $89 \%$, respectively. However, CollaborER produces positive and negative labels with an average accuracy of $99 \%$ and $97 \%$ respectively, as verified in Section 6.5.2. Compared to ERGAN, CollaborER gains up to $11 \%$ improvement on the quality of the generated labels.

### 6.4 Ablation Study

Next, we analyse the effectiveness of each proposed phase of CollaborER (i.e., ALG and CERT) by comparing CollaborER with its variants without the key optimization(s) in each phase. The results are shown in Figure 5, where the labels listed along the abscissa have the following meanings: (i) "CollaborER" represents its performance when all optimizations are used; (ii) "ALG" means the performance of CollaborER without (w/o) ALG; (iii) "CERT" denotes the performance of CollaborER w/o CERT; and (iv) "Train" " represents the performance of CollaborER w/o training.
CollaborER vs. CollaborER w/o ALG. ALG contains two components, i.e., RPLG and SNLG. Since CollaborER cannot work without RPLG, we focus on investigating the effectiveness of SNLG by replacing it with a random negative label generation method. It is observed that the F1-score drops $17.87 \%$ on average. This confirms that generating "challenging" negative labels based on semantic similarity greatly helps to train effective ER models. We also observe that the SNLG brings no more than $8.33 \%$ improvement on FZ dataset. This is attributed to the nature of this dataset, as it is relatively easier for CollaborER and all the competitors to achieve the perfect performance, i.e., $100 \%$ F1-score, in this dataset.
CollaborER vs. CollaborER w/o CERT. The difference between the proposed CERT and other existing ER models that also fine-tune pre-trained LMs lies in whether there is the intervention of graph features (w.r.t MRGFL) to assist the fine-tuning process. By removing MRGFL in CERT, the F1-score of CollaborER drops 3\% on average over the eight experimental datasets. Particularly, the drop of the F1-score is up to $5 \%$ on the DA-clean dataset. This shows that learning tuples' graph features is indispensable for promoting ER performance.
CollaborER vs. CollaborER w/o Train. We also explore the performance of CollaborER without any training process. In this case, CollaborER performs ER purely based on RPLG, which automatically discovers the matched tuples based on the semantic similarity. The results indicate that RPLG can find a large quantity of reliable matched tuples. It is worth noting that RPLG alone can achieve considerable results, e.g., $\sim 99 \%$ F1-score and $100 \%$ F1-score on DA-clean dataset and FZ dataset, respectively. This is because, matched tuples are mutually most similar with each other in those datasets. Since RPLG is general enough to perform ER in various datasets, it is possible to be widely used in practical ER applications without any time-consuming training process.

### 6.5 Further Experiments

We further justify the effectiveness of the proposed Colla$\operatorname{borER}$ by conducting the following two sets of experiments.

### 6.5.1 Graph Scale Analysis

The first set of experiments is to compare the scale of the graphs generated by the proposed MRGC and other graph construction methods in the existing ER approaches, i.e., EMBDI [5] and GraphER [23]. Figure 6 depicts the total number of nodes (denoted as \#Nodes) and that of edges (denoted as \#Edges) of graphs with regard to each dataset.
![img-5.jpeg](img-5.jpeg)

Fig. 6. Graph scale analysis
TABLE 4
The reliability analysis of ALG

| Datasets | RPLG |  |  | SNLG |  |  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  | TP | FN | TPR | TN | FP | TNR |
| AG | 332 | 0 | 1 | 7136 | 115 | 0.98 |
| BR | 34 | 0 | 1 | 17417 | 1074 | 0.94 |
| DA-clean | 2136 | 0 | 1 | 34119 | 3 | 0.99 |
| FZ | 102 | 0 | 1 | 1788 | 11 | 0.99 |
| IA-clean | 4 | 0 | 1 | 8399 | 521 | 0.94 |
| DA-dirty | 1941 | 0 | 1 | 30899 | 7 | 0.99 |
| IA-dirty | 2 | 0 | 1 | 7872 | 490 | 0.94 |
| AB | 247 | 2 | 0.99 | 4093 | 11 | 0.99 |

It is observed that MRGC generates much smaller graphs, compared against other graph generation methods. The reduced size of graphs greatly saves the memory for storing and training graphs, and reduces the training cost.

### 6.5.2 ALG Analysis

The second set of experiments is to verify the performance of ALG. To better study the quality of labels, we utilize six metrics: (i) true-positive (TP), which represents the number of truly labeled matched tuples; (ii) true-negative (TN), which denotes the number of truly labeled mismatched tuples; (iii) false-negative (FN), which represents the number of matched tuples that are labeled as mismatched; (iv) false-positive (FP), which denotes the number of mismatched tuples that are labeled as matched; (v) true-positive rate (TPR) represents the proportion of matched tuples that are correctly labeled, denoted as $\frac{T P}{T P+F N}$; and (vi) true-negative rate (TNR) represents the proportion of mismatched tuples that are correctly labeled, denoted as $\frac{T N}{T N+F P}$.
Analysis of label generating quality. We first evaluate the quality of the labels generated by ALG, including the positive labels generated by PRLG and the negative labels produced by SNLG. The results are reported in Table 4. As expected, both PRLG and SNLG are able to achieve the outstanding performance when generating labels. Specifically, CollaborER produces positive and negative labels with an average accuracy of $99 \%$ and $97 \%$, respectively. It confirms the effectiveness of our proposed ALG. The positive labels with high reliability allow the subsequent CERT model to be well-trained; while the generated negative labels enable CERT to identify "challenging" tuple pairs.
Effect of $\epsilon$-nearest neighbors for SNLG. We then study the performance of SNLG by varying $\epsilon$ among $\{10,50,90$, $|T|\}$. Note that, when $\epsilon=|T|$, SNLG generates negative labels by searching for possible tuples in the entire dataset.

![img-6.jpeg](img-6.jpeg)

Fig. 7. Effect of $\epsilon$-nearest neighbors for SNLG
![img-7.jpeg](img-7.jpeg)

Fig. 8. Demonstration. The detected anomalous values are highlighted in red cells.

In this case, SNLG behaves like random sampling. Figure 7 plots the corresponding results. We observe that F1-score of CollaborER drops as $\epsilon$ grows. This is because, the larger the $\epsilon$, the more likely the tuple pairs that are not similar to each other will be included in the set of negative labels. The dissimilar tuples contribute little to the training of an effective ER model, as discussed in Section 3. Besides, as expected, the quality of negative labels is still stable when $\epsilon$ changes, which could be observed from TNR (true-negative rate) values. This further demonstrates the effectiveness of the proposed SNLG.

## 7 DEMONSTRATION

Based on the proposed CollaborER, we further develop a cross-platform prototype system CollaborAD for anomaly detection. The basic interface is shown in Figure 8. The upper navigation bar illustrates the main components of the demo system, including the input component, the ER component, and the output component. We demonstrate CollaborAD using a real-world ER benchmark, i.e., Amazon-Google. At the input, a user receives two datasets
![img-8.jpeg](img-8.jpeg)
from Amazon-Google. After the system pre-processes the datasets, it proceeds to perform ER task. During ER, the system first generates positive labels and negative labels via RPLG and SNLG, respectively. Then, it constructs graphs for the inputs and learns the graph features of tuples according to the graph structures via MRGFL. Next, it employs the well-trained graph features of tuples to assist the CSFL model in discovering sufficient sentence features of tuples, and produces the final ER results. As depicted in Figure 8, the ER results and the identified anomalies are reported by the output component. It displays a set of matched tuples connected by double-head arrows. The anomalous values of those tuples are highlighted in red cells. In the event that there is no anomaly in any of the datasets, the system does not highlight anything.

## 8 Related Work

Entity Resolution (ER) is one of the fundamental and significant tasks in data curation. Early studies exploit rules [1], [12], [17], [37], [38] or crowdsourcing [14], [29], [44] for ER tasks. Rule-based solutions require human-provided declarative matching rules [17] or program-synthesized matching rules [38] to find matching pairs. Crowdsourcing-based solutions employ crowds to manually identify whether two tuples refer to the same real-world entity. Such solutions highly rely on human guidance, and have limitations in handling heterogeneous data. Recently, machine learning (ML) techniques have been widely used for ER and have achieved promising performance [9]. According to whether supervision signals are incorporated, existing ML-based solutions can be clustered into two categories, namely, supervised ER and unsupervised ER.

Supervised ER approaches [4], [5], [10], [13], [18], [21][23], [25], [32], [50], [51] can provide the state-of-the-art performance for ER, but require a substantial number of labels in the form of matches and mismatches, to support the learning of a reliable ER model. In general, the methods first learn the features of tuples via ML models and then feed the well-trained features into a binary classifier for identifying matched tuples.

A majority of supervised ER methods employ sentencebased ML model to learn the sentence features of tuples. DeepER [10] and DeepMatcher [32] utilize vanilla RNNs. MCA [50] proposes a multi-context attention mechanism to enrich the sentence features of tuples. Furthermore, current studies [4], [22], [25] indicate that applying pre-trained LMs to ER tasks achieves outstanding performance. DITTO [25] obtains the best performance among all the existing supervised ER works. It fine-tunes the pre-trained LMs with the help of a series of newly proposed data augmentation techniques. Several supervised ER methods transform a collection of tuples with the relational format to graph structures, and learn the graph features of tuples based on the constructed graphs [5], [23].

However, both sentence-based methods and graphbased methods are far from enough to capture sufficient features of tuples, as mentioned in Section 1. Our proposed CollaborER is introduced to enrich the features of tuples by learning both sentence features and graph features collaboratively. Besides, we have compared CollaborER with four state-of-the-art supervised ER solutions, and have verified that CollaborER, with zero labor-intensive labeling process, achieves comparable or even superior results, as compared with supervised approaches.

Unsupervised ER approaches [5], [46], [49] are designed to perform ER without labeling. ZeroER [46] learns the match and mismatch distributions based on Gaussian Mixture Models. EMBDI [5] performs ER by learning a compact graph-based representation for each tuple. ITER+CliqueRank [49] first constructs a bipartite graph to model the relationship between tuple pairs, and then develops an iterative-based ranking algorithm to estimate the similarity of tuple pairs. Despite the benefit of zero label requirement, unsupervised approaches are highly errorsensitive and may suffer from poor ER results when errors are contained in datasets. Considering that real-world datasets are often dirty, it is impractical to use the existing unsupervised ER methods in practice.

On the contrary, the proposed CollaborER performs ER in a self-supervised manner, which has the capability to perform ER in a fault-tolerant manner, as verified in the experiments reported in Section 6.3. In addition, we have compared CollaborER with two state-of-the-art unsupervised methods, including ZeroER and EMBDI. Note that we exclude ITER+CliqueRank from experiments since its performance is inferior to the two unsupervised methods that are selected as competitors in our study.

## 9 CONCLUSIONS

In this paper, we propose CollaborER, a self-supervised entity resolution framework, to perform the ER task with zero labor-intensive manual labeling. CollaborER conducts ER tasks by a pipe-lined modular architecture consisting of two phases, i.e., automatic label generation (ALG) and collaborative ER training (CERT). First, ALG is developed to automatically generate both reliable positive labels (w.r.t. RPLG) and semantic-based negative labels (w.r.t. SNLG). ALG is essential for the subsequent CERT phase since it provides high-quality labels that are the backbone of training effective ER models. Second, the framework proceeds to the CERT
phase, where tuples' sentence features and graph features are learned and employed collaboratively to produce the final ER results. In this phase, we first propose a multirelational graph construction (MRGC) method to construct graphs for each relational dataset, and then exploit GNN to learn the graph features of tuples. Thereafter, the welltrained graph features are fed into a collaborative sentence feature learning (CSFL) model to discover sufficient sentence features of tuples. Finally, CSFL predicts the matched tuple pairs and unmatched ones according to the learned features.

Currently, CollaborER treats the graph feature learning process (i.e., CSFL) as a black box, and uses graph features of tuples generated by AttrGNN [27] in the current implementation. In the near future, we plan to design a graph feature engineering solution for ER.

## REFERENCES

[1] A. Arasu, C. Ré, and D. Suciu. Large-scale deduplication with constraints using dedupalog. In ICDE, pages 952-963, 2009.
[2] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga. USAD: unsupervised anomaly detection on multivariate time series. In $K D D$, pages 3395-3404, 2020.
[3] A. Bojchevski and S. Günnemann. Adversarial attacks on node embeddings via graph poisoning. In ICML, pages 695-704, 2019.
[4] U. Brunner and K. Stockinger. Entity matching with transformer architectures - A step forward in data integration. In EDBT, pages $463-473,2020$.
[5] R. Cappuzzo, P. Papotti, and S. Thirumuruganathan. Creating embeddings of heterogeneous relational datasets for data integration tasks. In SIGMOD, pages 1335-1349, 2020.
[6] J. Chen, J. Zhu, and L. Song. Stochastic training of graph convolutional networks with variance reduction. In ICML, pages 941-949, 2018.
[7] P. Christen. Data Matching - Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection. Data-Centric Systems and Applications. Springer, 2012.
[8] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, pages 4171-4186, 2019.
[9] X. L. Dong and T. Rekatsinas. Data integration and machine learning: A natural synergy. In SIGMOD, pages 1645-1650, 2018.
[10] M. Ebraheem, S. Thirumuruganathan, S. R. Joty, M. Ouzzani, and N. Tang. Distributed representations of tuples for entity resolution. Proc. VLDB Endow., 11(11):1454-1467, 2018.
[11] W. Fan and F. Geerts. Foundations of Data Quality Management. Synthesis Lectures on Data Management. Morgan \& Claypool Publishers, 2012.
[12] W. Fan, X. Jia, J. Li, and S. Ma. Reasoning about record matching rules. Proc. VLDB Endow., 2(1):407-418, 2009.
[13] C. Fu, X. Han, L. Sun, B. Chen, W. Zhang, S. Wu, and H. Kong. End-to-end multi-perspective matching for entity resolution. In IJCAI, pages 4961-4967, 2019.
[14] C. Gokhale, S. Das, A. Doan, J. F. Naughton, N. Rampalli, J. W. Shavlik, and X. Zhu. Corleone: hands-off crowdsourcing for entity matching. In SIGMOD, pages 601-612, 2014.
[15] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In ICLR, 2015.
[16] W. L. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. In NeurIPS, pages 1024-1034, 2017.
[17] M. A. Hernández and S. J. Stolfo. The merge/purge problem for large databases. In SIGMOD, pages 127-138, 1995.
[18] J. Kasai, K. Qian, S. Gurajada, Y. Li, and L. Popa. Low-resource deep entity resolution with transfer and active learning. In ACL, pages 5851-5861, 2019.
[19] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2017.
[20] P. Konda, S. Das, P. S. G. C., A. Doan, A. Ardalan, J. R. Ballard, H. Li, F. Panahi, H. Zhang, J. F. Naughton, S. Prasad, G. Krishnan, R. Deep, and V. Raghavendra. Magellan: Toward building entity matching management systems. Proc. VLDB Endow., 9(12):11971208, 2016.

[21] H. Köpcke, A. Thor, and E. Rahm. Evaluation of entity resolution approaches on real-world match problems. Proc. VLDB Endow., 3(1):484-493, 2010.
[22] B. Li, Y. Miao, Y. Wang, Y. Sun, and W. Wang. Improving the efficiency and effectiveness for bert-based entity resolution. In $A A A I, 2020$.
[23] B. Li, W. Wang, Y. Sun, L. Zhang, M. A. Ali, and Y. Wang. Grapher: Token-centric entity resolution with graph convolutional neural networks. In $A A A I$, pages 8172-8179, 2020.
[24] C. Li, Y. Cao, L. Hou, J. Shi, J. Li, and T. Chua. Semi-supervised entity alignment via joint knowledge embedding model and crossgraph model. In EMNLP, pages 2723-2732, 2019.
[25] Y. Li, J. Li, Y. Suhara, A. Doan, and W. Tan. Deep entity matching with pre-trained language models. Proc. VLDB Endow., 14(1):5060, 2020.
[26] B. Liang, H. Li, M. Su, P. Bian, X. Li, and W. Shi. Deep text classification can be fooled. In IJCAI, pages 4208-4215, 2018.
[27] Z. Liu, Y. Cao, L. Pan, J. Li, and T. Chua. Exploring and evaluating attributes, values, and structures for entity alignment. In EMNLP, pages 6355-6364, 2020.
[28] X. Mao, W. Wang, H. Xu, Y. Wu, and M. Lan. Relational reflection entity alignment. In CIKM, pages 1095-1104, 2020.
[29] A. Marcus, E. Wu, D. R. Karger, S. Madden, and R. C. Miller. Human-powered sorts and joins. Proc. VLDB Endow., 5(1):13-24, 2011.
[30] L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein. Unrolled generative adversarial networks. In $I C L R, 2017$.
[31] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In NeurIPS, pages 3111-3119, 2013.
[32] S. Mudgal, H. Li, T. Rekatsinas, A. Doan, Y. Park, G. Krishnan, R. Deep, E. Arcaute, and V. Raghavendra. Deep learning for entity matching: A design space exploration. In SIGMOD, pages 19-34, 2018.
[33] S. Mudgal, H. Li, T. Rekatsinas, A. Doan, Y. Park, G. Krishnan, R. Deep, E. Arcaute, and V. Raghavendra. Deep learning for entity matching: A design space exploration. In SIGMOD, pages 19-34, 2018.
[34] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, pages 8024-8035, 2019.
[35] N. Reimers and I. Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In EMNLP-IJCNLP, pages 39803990, 2019.
[36] J. Shao, Q. Wang, A. Wijesinghe, and E. Rahm. Ergan: Generative adversarial networks for entity resolution. In ICDM, pages 12501255, 2020.
[37] R. Singh, V. V. Meduri, A. K. Elmagarmid, S. Madden, P. Papotti, J. Quiané-Ruiz, A. Solar-Lezama, and N. Tang. Generating concise entity matching rules. In SIGMOD, pages 1635-1638, 2017.
[38] R. Singh, V. V. Meduri, A. K. Elmagarmid, S. Madden, P. Papotti, J. Quiané-Ruiz, A. Solar-Lezama, and N. Tang. Synthesizing entity matching rules by examples. Proc. VLDB Endow., 11(2):189-202, 2017.
[39] Z. Sun, Z. Deng, J. Nie, and J. Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. In $I C L R, 2019$.
[40] Z. Sun, W. Hu, Q. Zhang, and Y. Qu. Bootstrapping entity alignment with knowledge graph embedding. In IJCAI, pages 4396-4402, 2018.
[41] Z. Sun, Q. Zhang, W. Hu, C. Wang, M. Chen, F. Akrami, and C. Li. A benchmarking study of embedding-based entity alignment for knowledge graphs. PVLDB, 13(11):2326-2340, 2020.
[42] T. Trouillon, J. Welbl, S. Riedel, É. Gaussier, and G. Bouchard. Complex embeddings for simple link prediction. In ICML, volume 48, pages 2071-2080, 2016.
[43] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. Graph attention networks. In ICLR, 2018.
[44] J. Wang, T. Kraska, M. J. Franklin, and J. Feng. Crowder: Crowdsourcing entity resolution. Proc. VLDB Endow., 5(11):1483-1494, 2012.
[45] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, and J. Brew. Huggingface's transformers: State-of-the-art natural language processing. CoRR, abs/1910.03771, 2019.
[46] R. Wu, S. Chaba, S. Sawlani, X. Chu, and S. Thirumuruganathan. Zeroer: Entity resolution using zero labeled examples. In SIGMOD, pages 1149-1164, 2020.
[47] Z. Yang, Z. Dai, Y. Yang, J. G. Carbonell, R. Salakhutdinov, and Q. V. Le. Xlnet: Generalized autoregressive pretraining for language understanding. In NeurIPS, pages 5754-5764, 2019.
[48] W. Zeng, X. Zhao, W. Wang, J. Tang, and Z. Tan. Degree-aware alignment for entities in tail. In SIGIR, pages 811-820, 2020.
[49] D. Zhang, D. Li, L. Guo, and K.-L. Tan. Unsupervised entity resolution with blocking and graph algorithms. TKDE, 2020.
[50] D. Zhang, Y. Nie, S. Wu, Y. Shen, and K. Tan. Multi-context attention for entity matching. In WWW, pages 2634-2640. ACM / IW3C2, 2020.
[51] C. Zhao and Y. He. Auto-em: End-to-end fuzzy entity-matching using pre-trained deep models and transfer learning. In WWW, pages 2413-2424, 2019.
[52] D. Zheng, C. Ma, M. Wang, J. Zhou, Q. Su, X. Song, Q. Gan, Z. Zhang, and G. Karypis. Distdgl: Distributed graph neural network training for billion-scale graphs. CoRR, abs/2010.05337, 2020.
[53] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen. Deep autoencoding gaussian mixture model for unsupervised anomaly detection. In ICLR, 2018.

