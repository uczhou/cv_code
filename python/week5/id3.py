import numpy as np
import pandas as pd


class Node:

    def __init__(self, key, val, result, children):
        self.key = key
        self.val = val
        self.result = result
        self.children = children


class ID3:

    def __init__(self, data_set):
        self.data_set = data_set
        self.root = self.data_split(data_set, [], None, None)

    @staticmethod
    def calc_entropy(num, total):
        # -p * logp
        p = float(num) / total
        return -p * np.log(p)

    def calc_info_gain(self, entropy, data_set, col_name):
        group_count = data_set.groupby(col_name).size().reset_index(name='counts')
        sub_group_count = data_set.groupby([col_name, 'label']).size().reset_index(name='counts')

        total_data = data_set.shape[0]

        col_entropy = 0
        for i in range(group_count.shape[0]):
            pos = sub_group_count.loc[
                sub_group_count[col_name] == group_count.iloc[i][col_name] & sub_group_count['label'] == 1]['counts']
            neg = sub_group_count.loc[
                sub_group_count[col_name] == group_count.iloc[i][col_name] & sub_group_count['label'] == 0]['counts']
            total = sub_group_count.iloc[i]['counts']
            group_entropy = 0
            if pos > 0:
                group_entropy += self.calc_entropy(pos, total)

            if neg > 0:
                group_entropy += self.calc_entropy(neg, total)

            col_entropy += (total / total_data) * group_entropy

        return entropy - col_entropy

    def attribute_selection(self, data_set, cols):
        # calculate entropy of data_set
        mask_ones = data_set['label'] == 1
        mask_zeros = data_set['label'] == 0

        entropy = self.calc_entropy(len(mask_ones), data_set.shape[0]) + self.calc_entropy(len(mask_zeros), data_set.shape[0])

        # calculate information gain
        info_gains = {}
        for col in data_set.columns:
            if col not in cols:
                info_gains = {col: self.calc_info_gain(entropy, data_set[[col, 'label']], col)}

        sorted_info_gains = sorted(info_gains.keys(), key=lambda x: -x[1])

        # get the most information gain
        return sorted_info_gains[0][0]

    def data_split(self, data_set, cols, key, val):
        if len(cols) == data_set.shape[1] - 1:
            return None

        if len(cols) == 0:
            root = Node(key, val, None, [])

            col = self.attribute_selection(data_set, cols)
            cols.append(col)

            groups = data_set[col].drop_duplicates().values.tolist()
            for group in groups:
                mask = data_set[col] == group
                if mask.size > 0:
                    child = self.data_split(data_set[mask], cols, col, group)
                    if child:
                        root.children.append(child)

            return root

        node = Node(key, val, None, [])

        mask_ones = data_set['label'] == 1
        mask_zeros = data_set['label'] == 0

        col = self.attribute_selection(data_set, cols)
        cols.append(col)

        if len(mask_ones) == data_set.shape[0]:
            node.result = True
            return node
        elif len(mask_zeros) == data_set.shape[0]:
            node.result = False
            return node

        groups = data_set[col].drop_duplicates().values.tolist()
        for group in groups:
            mask = data_set[col] == group
            if mask.size > 0:
                child = self.data_split(data_set[mask], cols, col, group)
                if child:
                    node.children.append(child)

        return node

    def predict(self, node, data):

        for child in node.children:
            if data[child.val] == data[child.key]:
                if child.result:
                    return child.result
                else:
                    return self.predict(child, data)

        return False



