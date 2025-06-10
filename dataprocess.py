class ProcessData:
    def __init__(self):
        load_male=open('dataset/male.txt','r',encoding="UTF-8")
        load_female=open('dataset/female.txt','r',encoding="UTF-8")
        load_stem=open('dataset/stem.txt','r',encoding="UTF-8")
        load_family=open('dataset/family.txt','r',encoding="UTF-8")
        load_career=open('dataset/career.txt','r',encoding="UTF-8")
        load_arts=open('dataset/arts.txt','r',encoding='UTF-8')
        load_intelligence=open('dataset/intelligence.txt','r',encoding='UTF-8')
        load_strong=open('dataset/strong.txt','r',encoding='UTF-8')
        load_weak=open('dataset/weak.txt','r',encoding='UTF-8')
        load_shape=open('dataset/shape.txt','r',encoding='UTF-8')
 
        self.male_words=[word.replace('\n','') for word in load_male.readlines()]
        self.female_words=[word.replace('\n','') for word in load_female.readlines()]
        self.stem_words=[word.replace('\n','') for word in load_stem.readlines()]
        self.family_words=[word.replace('\n','') for word in load_family.readlines()]
        self.career_words=[word.replace('\n','') for word in load_career.readlines()]
        self.strong_words=[word.replace('\n','') for word in load_strong.readlines()]
        self.weak_words=[word.replace('\n','') for word in load_weak.readlines()]
        self.load_arts=[word.replace('\n','') for word in load_arts.readlines()]
        self.load_intelligence=[word.replace('\n','') for word in load_intelligence.readlines()]
        self.shape_words=[word.replace('\n','') for word in load_shape.readlines()]

        self.word_store={
            "male_terms":self.male_words,
            "female_terms":self.female_words,
            "stem_words":self.stem_words,
            "family_words":self.family_words,
            "career_words":self.career_words,
            "intelligence_words":self.load_intelligence,
            "weak_words":self.weak_words,
            "strong_words":self.strong_words,
            "shape_words":self.shape_words
        }


        load_male.close()
        load_female.close()
        load_stem.close()
        load_family.close()
        load_career.close()
        load_intelligence.close()
        load_arts.close()
        load_strong.close()
        load_weak.close()
        load_shape.close()

    def get_attributes(self,dataset_name:str):
        if dataset_name in self.word_store:
            return self.word_store[dataset_name]
        else:
            raise Exception(f"List of dataset names are {','.join(list(self.word_store.keys()))}")
