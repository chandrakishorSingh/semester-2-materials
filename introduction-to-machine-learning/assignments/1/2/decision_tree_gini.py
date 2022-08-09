import pandas as pd
import math


class DTNode:
    def __init__(self, node_type="label", label=None, gini_index=None, feature_name=None, left_attr_collection=None,
                 right_attr_collection=None, left=None, right=None):
        self.node_type = node_type
        self.label = label
        self.gini_index = gini_index
        self.feature_name = feature_name
        self.left_attr_collection = left_attr_collection  # this will be a `list` type
        self.right_attr_collection = right_attr_collection
        self.left = left
        self.right = right


class DT:
    def __init__(self, data, target_feature):
        self.data = data
        self.target_feature = target_feature
        self.features = self._get_features()
        remaining_features = self.features.copy()
        del remaining_features[self.target_feature]
        self.root = self._build_tree(self.data.copy(), remaining_features.copy())

    def _get_feature_attr_sets(self, attrs):
        attr_sets = []
        set_count = 2 ** (len(attrs) - 1) - 1
        for i in range(1, set_count + 1):
            attr_set = []
            for j in range(len(attrs)):
                if (1 << j) & i:
                    attr_set.append(attrs[j])
            attr_sets.append([attr_set, list(set(attrs) - set(attr_set))])

        return attr_sets

    def predict(self, record):
        decision_path = []
        node = self.root

        while node.node_type != 'label':
            decision_path.append(str(node.feature_name) + '=' + str(record[node.feature_name]))
            if record[node.feature_name] in node.left_attr_collection:
                node = node.left
            else:
                node = node.right

        decision_path.append(f"[prediction={node.label}]")

        return decision_path

    def _get_features(self):
        features = {}

        feature_names = list(self.data.columns)
        for feature in feature_names:
            features[feature] = set(self.data.get(feature))

        return features

    def _build_tree(self, data, features):
        if len(features) == 0:
            target_attr_value_list = list(data.get(self.target_feature))
            target_attr_value_count = {}
            for attr_value in target_attr_value_list:
                if attr_value in target_attr_value_count:
                    target_attr_value_count[attr_value] = target_attr_value_count[attr_value] + 1
                else:
                    target_attr_value_count[attr_value] = 1

            max_attr_value_count = 0
            max_attr = ''
            for attr_value in target_attr_value_count:
                if max_attr_value_count < target_attr_value_count[attr_value]:
                    max_attr_value_count = target_attr_value_count[attr_value]
                    max_attr = attr_value

            return DTNode(label=max_attr)

        if len(set(data.get(self.target_feature))) == 1:
            return DTNode(label=data.get(self.target_feature).iloc[0])

        feature_gini_indices = {}
        feature_attr_set = {}

        for feature_name in features:
            attr_sets = self._get_feature_attr_sets(list(features[feature_name]))
            for attr_set in attr_sets:
                subset_data_left = pd.DataFrame()
                for attr in attr_set[0]:
                    subset_data_left = subset_data_left.append(data.loc[data[feature_name] == attr])

                subset_data_right = pd.DataFrame()
                for attr in attr_set[1]:
                    subset_data_right = subset_data_right.append(data.loc[data[feature_name] == attr])

                gini_index_left = self._get_gini_index(subset_data_left)
                gini_index_right = self._get_gini_index(subset_data_right)

                n1 = len(subset_data_left)
                n2 = len(subset_data_right)
                gini_index = (n1 / (n1 + n2)) * gini_index_left + (n2 / (n1 + n2)) * gini_index_right

                if feature_name in feature_gini_indices:
                    if feature_gini_indices[feature_name] > gini_index:
                        feature_gini_indices[feature_name] = gini_index
                        feature_attr_set[feature_name] = attr_set
                else:
                    feature_gini_indices[feature_name] = gini_index
                    feature_attr_set[feature_name] = attr_set

        min_gini_index = 1
        min_gini_index_feature = ''
        min_gini_index_attr_set = []
        for feature_name in feature_gini_indices:
            if feature_gini_indices[feature_name] < min_gini_index:
                min_gini_index = feature_gini_indices[feature_name]
                min_gini_index_feature = feature_name
                min_gini_index_attr_set = feature_attr_set[feature_name]

        left_tree_data = pd.DataFrame()
        for attr in min_gini_index_attr_set[0]:
            left_tree_data = left_tree_data.append(data.loc[data[min_gini_index_feature] == attr])

        right_tree_data = pd.DataFrame()
        for attr in min_gini_index_attr_set[1]:
            right_tree_data = right_tree_data.append(data.loc[data[min_gini_index_feature] == attr])

        subtree_tree_features = features.copy()
        del subtree_tree_features[min_gini_index_feature]

        node = DTNode(node_type='internal', label=None, gini_index=min_gini_index, feature_name=min_gini_index_feature,
                      left_attr_collection=min_gini_index_attr_set[0],
                      right_attr_collection=min_gini_index_attr_set[1])

        node.left = self._build_tree(left_tree_data.copy(), subtree_tree_features.copy())
        node.right = self._build_tree(right_tree_data.copy(), subtree_tree_features.copy())

        return node

    def _get_gini_index(self, data):
        label_counts = {}

        for label in data.get(self.target_feature):
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        sample_count = sum([label_counts[label] for label in label_counts])
        probability_sqr_sum = sum([(label_counts[label] / sample_count) ** 2 for label in label_counts])

        return 1 - probability_sqr_sum


data = pd.read_csv('data.csv')

dt = DT(data, 'profit')

print(dt.predict({'price': 'low', 'maintenance': 'high', 'capacity': 5, 'airbag': 'no'}))
