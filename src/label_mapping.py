import pickle

class LabelMapping:
    def __init__(self):
        self.mapping = {}

    @staticmethod
    def convert_to_tuple(tokenized_label):
        return tuple(map(tuple, [tuple(sub_list) for sub_list in tokenized_label]))

    @staticmethod
    def convert_from_tuple(flat_tuple):
        return tuple(map(tuple, [list(sub_tuple) for sub_tuple in flat_tuple]))

    def add_mapping(self, original_label, tokenized_label):
        tokenized_label = tokenized_label[0]
        # print(tokenized_label)
        tokenized_label = self.convert_to_tuple(tokenized_label[0])
        valid_original_label = self.convert_to_tuple(original_label)
        # print(tokenized_label, valid_original_label)
        #may also need to convert_to_tuple the original label to store it (actually no)
        self.mapping[tokenized_label] = valid_original_label

    def get_original_label(self, tokenized_label):
        tokenized_label = self.convert_to_tuple(tokenized_label)
        return self.mapping.get(tokenized_label, None)

    #TODO: leave this for reference for now
    # def add_mapping(self, tokenized_sentence, label_mapping_file):
        #tokenized_label is stored as list but that isn't hashable so dump contents out
        # tokenized_label = self.convert_to_tuple(tokenized_label)
        # self.mapping[tokenized_label] = original_label
        # print(tokenized_sentence[0])
        # self.mapping[tuple(tokenized_sentence[0])] = label_mapping_file

    # def get_label_mapping(self, tokenized_sentence):
    #     return self.mapping[tuple(tokenized_sentence)]

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.mapping, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.mapping = pickle.load(f)


