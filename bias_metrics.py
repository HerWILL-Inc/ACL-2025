import torch

class WordEmbeddingAssociationTest:
    """
    Word Embedding Association Test (WEAT) score measures the unlikelihood there is no
    difference between two sets of target words in terms of their relative similarity
    to two sets of attribute words by computing the probability that a random
    permutation of attribute words would produce the observed (or greater) difference
    in sample means. Analog of Implicit Association Test from psychology for word embeddings.

    Based on: Caliskan, A., Bryson, J., & Narayanan, A. (2017). [Semantics derived automatically
    from language corpora contain human-like biases](https://api.semanticscholar.org/CorpusID:23163324).
    Science, 356, 183 - 186.
    """

    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings1: torch.Tensor,
        attribute_embeddings2: torch.Tensor,
        test_stat=False
    ) -> torch.FloatTensor:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        target_embeddings1 : `torch.Tensor`, required.
            A tensor of size (target_embeddings_batch_size, ..., dim) containing target word
            embeddings related to a concept group. For example, if the concept is gender,
            target_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc. Represented as X.

        target_embeddings2 : `torch.Tensor`, required.
            A tensor of the same size as target_embeddings1 containing target word
            embeddings related to a different group for the same concept. For example,
            target_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc. Represented as Y.

        attribute_embeddings1 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings1_batch_size, ..., dim) containing attribute word
            embeddings related to a concept group associated with the concept group for target_embeddings1.
            For example, if the concept is professions, attribute_embeddings1 could contain embeddings for
            stereotypically male professions, e.g. "doctor", "banker", "engineer", etc. Represented as A.

        attribute_embeddings2 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings2_batch_size, ..., dim) containing attribute word
            embeddings related to a concept group associated with the concept group for target_embeddings2.
            For example, if the concept is professions, attribute_embeddings2 could contain embeddings for
            stereotypically female professions, e.g. "nurse", "receptionist", "homemaker", etc. Represented as B.

        !!! Note
            While target_embeddings1 and target_embeddings2 must be the same size, attribute_embeddings1 and
            attribute_embeddings2 need not be the same size.

        # Returns

        weat_score : `torch.FloatTensor`
            The unlikelihood there is no difference between target_embeddings1 and target_embeddings2 in
            terms of their relative similarity to attribute_embeddings1 and attribute_embeddings2.
            Typical values are around [-1, 1], with values closer to 0 indicating less biased associations.

        """

        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise Exception(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings1.ndim < 2 or attribute_embeddings2.ndim < 2:
            raise Exception(
                "attribute_embeddings1 and attribute_embeddings2 must have at least two dimensions."
            )
        # if target_embeddings1.size() != target_embeddings2.size():
        #     raise Exception(
        #         "target_embeddings1 and target_embeddings2 must be of the same size."
        #     )
        # if attribute_embeddings1.size(dim=-1) != attribute_embeddings2.size(dim=-1) or attribute_embeddings1.size(dim=-1) != target_embeddings1.size(dim=-1):
        #     raise Exception(
        #         "All embeddings must have the same dimensionality."
        #     )


        target_embeddings1 = target_embeddings1.flatten(end_dim=-2)
        target_embeddings2 = target_embeddings2.flatten(end_dim=-2)
        attribute_embeddings1 = attribute_embeddings1.flatten(end_dim=-2)
        attribute_embeddings2 = attribute_embeddings2.flatten(end_dim=-2)

        # Normalize
        target_embeddings1 = torch.nn.functional.normalize(target_embeddings1, p=2, dim=-1)
        target_embeddings2 = torch.nn.functional.normalize(target_embeddings2, p=2, dim=-1)
        attribute_embeddings1 = torch.nn.functional.normalize(attribute_embeddings1, p=2, dim=-1)
        attribute_embeddings2 = torch.nn.functional.normalize(attribute_embeddings2, p=2, dim=-1)

        # Compute cosine similarities
        X_sim_A = torch.mm(target_embeddings1, attribute_embeddings1.t())
        X_sim_B = torch.mm(target_embeddings1, attribute_embeddings2.t())
        Y_sim_A = torch.mm(target_embeddings2, attribute_embeddings1.t())
        Y_sim_B = torch.mm(target_embeddings2, attribute_embeddings2.t())
        X_union_Y_sim_A = torch.cat([X_sim_A, Y_sim_A])
        X_union_Y_sim_B = torch.cat([X_sim_B, Y_sim_B])

        s_X_A_B = torch.mean(X_sim_A, dim=-1) - torch.mean(X_sim_B, dim=-1)
        s_Y_A_B = torch.mean(Y_sim_A, dim=-1) - torch.mean(Y_sim_B, dim=-1)
        s_X_Y_A_B = torch.mean(s_X_A_B) - torch.mean(s_Y_A_B)
        S_X_union_Y_A_B = torch.mean(X_union_Y_sim_A, dim=-1) - torch.mean(X_union_Y_sim_B, dim=-1)
        if test_stat:
            return s_X_Y_A_B
        else:
            return s_X_Y_A_B / torch.std(S_X_union_Y_A_B, unbiased=False)
    

class EmbeddingCoherenceTest:
    """
    Embedding Coherence Test (ECT) score measures if groups of words
    have stereotypical associations by computing the Spearman Coefficient
    of lists of attribute embeddings sorted based on their similarity to
    target embeddings.

    Based on: Dev, S., & Phillips, J.M. (2019). [Attenuating Bias in Word Vectors]
    (https://api.semanticscholar.org/CorpusID:59158788). AISTATS.
    """

    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings: torch.Tensor,
    ) -> torch.FloatTensor:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        target_embeddings1 : `torch.Tensor`, required.
            A tensor of size (target_embeddings_batch_size, ..., dim) containing target word
            embeddings related to a concept group. For example, if the concept is gender,
            target_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc. Represented as X.

        target_embeddings2 : `torch.Tensor`, required.
            A tensor of the same size as target_embeddings1 containing target word
            embeddings related to a different group for the same concept. For example,
            target_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc. Represented as Y.

        attribute_embeddings : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings_batch_size, ..., dim) containing attribute word
            embeddings related to a concept associated with target_embeddings1 and target_embeddings2.
            For example, if the concept is professions, attribute_embeddings could contain embeddings for
            "doctor", "banker", "engineer", etc. Represented as AB.

        # Returns

        ect_score : `torch.FloatTensor`
            The Spearman Coefficient measuring the similarity of lists of attribute embeddings sorted
            based on their similarity to the target embeddings. Ranges from [-1, 1], with values closer
            to 1 indicating less biased associations.

        """
        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise Exception(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings.ndim < 2:
            raise Exception("attribute_embeddings must have at least two dimensions.")
        # if target_embeddings1.size() != target_embeddings2.size():
        #     raise Exception(
        #         "target_embeddings1 and target_embeddings2 must be of the same size."
        #     )
        # if attribute_embeddings.size(dim=-1) != target_embeddings1.size(dim=-1):
        #     raise Exception("All embeddings must have the same dimensionality.")

        mean_target_embedding1 = target_embeddings1.flatten(end_dim=-2).mean(dim=0)
        mean_target_embedding2 = target_embeddings2.flatten(end_dim=-2).mean(dim=0)
        attribute_embeddings = attribute_embeddings.flatten(end_dim=-2)

        # Normalize
        mean_target_embedding1 = torch.nn.functional.normalize(mean_target_embedding1, p=2, dim=-1)
        mean_target_embedding2 = torch.nn.functional.normalize(mean_target_embedding2, p=2, dim=-1)
        attribute_embeddings = torch.nn.functional.normalize(attribute_embeddings, p=2, dim=-1)

        # Compute cosine similarities
        AB_sim_m = torch.matmul(attribute_embeddings, mean_target_embedding1)
        AB_sim_f = torch.matmul(attribute_embeddings, mean_target_embedding2)

        return self.spearman_correlation(AB_sim_m, AB_sim_f)

    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(x.size(0), device=ranks.device)
        return ranks

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor):
        x_rank = self._get_ranks(x)
        y_rank = self._get_ranks(y)

        n = x.size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n**2 - 1.0)
        return 1.0 - (upper / down)
    
class CosMetric:

    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings: torch.Tensor,
    ) -> torch.FloatTensor:
    
        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise Exception(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings.ndim < 2:
            raise Exception("attribute_embeddings must have at least two dimensions.")
        # if target_embeddings1.size() != target_embeddings2.size():
        #     raise Exception(
        #         "target_embeddings1 and target_embeddings2 must be of the same size."
        #     )
        if attribute_embeddings.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise Exception("All embeddings must have the same dimensionality.")

        mean_target_embedding1 = target_embeddings1.flatten(end_dim=-2).mean(dim=0)
        mean_target_embedding2 = target_embeddings2.flatten(end_dim=-2).mean(dim=0)
        attribute_embeddings = attribute_embeddings.flatten(end_dim=-2)

        # Normalize
        mean_target_embedding1 = torch.nn.functional.normalize(mean_target_embedding1, p=2, dim=-1)
        mean_target_embedding2 = torch.nn.functional.normalize(mean_target_embedding2, p=2, dim=-1)
        attribute_embeddings = torch.nn.functional.normalize(attribute_embeddings, p=2, dim=-1)

        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)

        # Compute cosine similarities
        AB_sim_m = cos(attribute_embeddings, mean_target_embedding1)
        AB_sim_f = cos(attribute_embeddings, mean_target_embedding2)

        cos_m = AB_sim_m.mean(dim=0)
        cos_f = AB_sim_f.mean(dim=0)

        return cos_m,cos_f
    
class RND:

    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings: torch.Tensor,
    ) -> torch.FloatTensor:
    
        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise Exception(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings.ndim < 2:
            raise Exception("attribute_embeddings must have at least two dimensions.")
        # if target_embeddings1.size() != target_embeddings2.size():
        #     raise Exception(
        #         "target_embeddings1 and target_embeddings2 must be of the same size."
        #     )
        if attribute_embeddings.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise Exception("All embeddings must have the same dimensionality.")

        mean_target_embedding1 = target_embeddings1.flatten(end_dim=-2).mean(dim=0)
        mean_target_embedding2 = target_embeddings2.flatten(end_dim=-2).mean(dim=0)
        attribute_embeddings = attribute_embeddings.flatten(end_dim=-2)

        # Normalize
        mean_target_embedding1 = torch.nn.functional.normalize(mean_target_embedding1, p=2, dim=-1)
        mean_target_embedding2 = torch.nn.functional.normalize(mean_target_embedding2, p=2, dim=-1)
        attribute_embeddings = torch.nn.functional.normalize(attribute_embeddings, p=2, dim=-1)

        print("before norming: ", mean_target_embedding1.shape)

        T1 = torch.linalg.matrix_norm(attribute_embeddings-mean_target_embedding1,ord='fro')
        T2 = torch.linalg.matrix_norm(attribute_embeddings-mean_target_embedding2,ord='fro')

        print("after norming: ",T1.shape)

        rnd = torch.sum(T1-T2)

        print("rnd shape: ",rnd.shape) 

        return rnd
    
class RIPA:
    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings: torch.Tensor,
    ) -> torch.FloatTensor:
        def _b_vec(word_vec_1, word_vec_2):
            # calculating the relation vector
            vec = word_vec_1 - word_vec_2
            norm = torch.norm(vec)
            
            vec/=norm
           
            return vec

        def _ripa_calc(word_vec, b_vec):
            # calculating the dot product of the relation vector with the attribute
            # word vector
            return torch.matmul(word_vec,b_vec.reshape(b_vec.shape[0],-1).T)
        
    
        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise Exception(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings.ndim < 2:
            raise Exception("attribute_embeddings must have at least two dimensions.")
        # if target_embeddings1.size() != target_embeddings2.size():
        #     raise Exception(
        #         "target_embeddings1 and target_embeddings2 must be of the same size."
        #     )
        if attribute_embeddings.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise Exception("All embeddings must have the same dimensionality.")
      
         # calculating the ripa score for each attribute word with each target pair
        bvec=_b_vec(target_embeddings1,target_embeddings2)
      
        ripa_calc=_ripa_calc(attribute_embeddings,bvec)
        ripa_calc=ripa_calc.reshape(ripa_calc.shape[0],-1)
        ripa_oa_mean=torch.mean(ripa_calc,dim=1)
        
        return torch.mean(ripa_oa_mean)
