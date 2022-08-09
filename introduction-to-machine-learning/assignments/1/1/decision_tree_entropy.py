import pandas as pd
import math


class DTNode:
    def __init__(self, node_type="label", label=None, entropy=None, feature_name=None, children=None, parent=None):
        self.node_type = node_type
        self.label = label
        self.entropy = entropy
        self.feature_name = feature_name
        self.children = children
        self.parent = parent


class DT:
    def __init__(self, data, target_feature):
        self.data = data
        self.target_feature = target_feature
        self.features = self._get_features()
        remaining_features = self.features.copy()
        del remaining_features[self.target_feature]
        self.root = self._build_tree(self.data.copy(), remaining_features.copy(), None)

    def predict(self, record):
        decision_path = []
        node = self.root

        while node.node_type != 'label':
            decision_path.append(str(node.feature_name) + '=' + str(record[node.feature_name]))
            node = node.children[record[node.feature_name]]

        decision_path.append(f"[prediction={node.label}]")

        return decision_path

    def _get_features(self):
        features = {}

        feature_names = list(self.data.columns)
        for feature in feature_names:
            features[feature] = set(self.data.get(feature))

        return features

    def _build_tree(self, data, features, parent):
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

        feature_entropies = {}

        for feature_name in features:
            attr_entropies = {}
            attr_record_count = {}
            attrs = features[feature_name]
            for attr in attrs:
                subset_data = data.loc[data[feature_name] == attr]
                attr_entropies[attr] = self._get_entropy(subset_data)
                attr_record_count[attr] = len(subset_data)

            feature_entropies[feature_name] = sum(
                [attr_entropies[attr] * attr_record_count[attr] for attr in attrs]) / sum(
                attr_record_count[attr] for attr in attrs)

        parent_entropy = self._get_entropy(self.data) if parent == None else parent.entropy

        igs = {feature_name: parent_entropy - feature_entropies[feature_name] for feature_name in features}

        max_ig_feature = ''
        max_ig = -1
        for feature_name in igs:
            if max_ig < igs[feature_name]:
                max_ig = igs[feature_name]
                max_ig_feature = feature_name

        node = DTNode(node_type='internal', entropy=feature_entropies[max_ig_feature], feature_name=max_ig_feature,
                      children=None, parent=parent)

        # distribute the data according to attr. of chosen feature and assign child nodes
        children = {}
        remaining_features = features.copy()
        del remaining_features[max_ig_feature]
        for attr in features[max_ig_feature]:
            remaining_data = data.copy()
            remaining_data = remaining_data.loc[remaining_data[max_ig_feature] == attr]
            children[attr] = self._build_tree(remaining_data.copy(), remaining_features.copy(), node)

        node.children = children

        return node

    def _get_entropy(self, data):
        label_counts = {}

        for label in data.get(self.target_feature):
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        sample_count = sum([label_counts[label] for label in label_counts])
        entropy = sum([
            (label_counts[label] / sample_count) * math.log((label_counts[label] / sample_count), 2)
            if label_counts[label] != 0 else 0 for label in label_counts
        ])

        return -1 * entropy



data = pd.read_csv('data.csv')

dt = DT(data, 'profit')

print(dt.predict({'price': 'low', 'maintenance': 'high', 'capacity': 5, 'airbag': 'no'}))

