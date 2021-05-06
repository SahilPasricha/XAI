import tensorflow as tf
from itertools import chain
import pandas as pd
import numpy as np
from tools.distance_calc import dynamic_time_warping
from tools.cluster_weights import aggomerativeClustering
from tools.cluster_weights import groupByClusters
from tools.Regression import SimpLinearRegression
from tools.uncluster_weights import uncluster_reg_weights


file_name_prefix = "Weights"
file_name_suffix = ".csv"
cluster_id_verbose = "Cluster_Id"
clus_reg_file_name_prefix = "clus_reg_"
unclus_reg_file_name_prefix = "unclus_reg_"

class StoreWeights(tf.keras.callbacks.Callback):
    def __init__(self,
                 filepath,
                 reg_train_steps,
                 dtw_clusters,
                 file_prefix,
                 skip_array = [],
                 weight_pred_ind = False,
                 weighs_dtw_cluster_ind = False,
                 save_freq = 1,
                 **kwargs):

        #super(StoreWeights, self).__init__()
        self.file_path = filepath
        self.reg_train_steps = reg_train_steps
        self.dtw_clusters = dtw_clusters
        self.skip_array = skip_array
        self.weight_pred_ind = weight_pred_ind
        self.weighs_dtw_cluster_ind = weighs_dtw_cluster_ind
        self.save_freq = save_freq
        self.file_name_prefix = file_prefix


    #def store_weights(self, file_path, reg_train_steps, dtw_clusters, skip_array, weight_pred_ind, weighs_dtw_cluster_ind):
    def store_weights(self, epoch):
        print("\n ***********Working for epoch ", epoch)
        skip_point_list = [i for i,j in self.skip_array]
        all_layer_weights = self.model.get_weights()

        # jump 2 steps to avoid odd layer--> 784* 128,128, 1280 ,<-- odd layers store 128 ,that's not weighst actually
        for i in range(0,len(self.model.layers),2):
            file_name = self.file_name_prefix + str(i).zfill(2) + file_name_suffix

            #only suited for 2d list
            layer_weights = self.model.get_weights()[i]
            layer_weights_flatten_raw = list(chain.from_iterable(layer_weights))
            layer_weights_flatten = [round(num, 3) for num in layer_weights_flatten_raw]
            #ToDo make it to 4 precision

            file_path_name = self.file_path + "/" + file_name
            layer_weights_length = self.model.get_weights()[i].shape[0]
            layer_weights_width = self.model.get_weights()[i].shape[1]

            if epoch == 0:
                Ids = [*range(len(layer_weights_flatten))]
                df = pd.DataFrame({'Id': Ids})
                df['Epoch:' + str(epoch)] = layer_weights_flatten
                df.to_csv(file_path_name, index = False, header=True,float_format='%.3f')

            else:
                df = pd.read_csv(file_path_name)
                custom_epoch = int(df.columns[-1].replace('Epoch:',''))+1
                df['Epoch:' + str(custom_epoch)] = layer_weights_flatten
                df.to_csv(file_path_name, index = False, header=True,float_format='%.3f')


            if epoch in skip_point_list and self.weight_pred_ind:
                # check if current epoch is skip point
                reg_prd_steps = [j for i,j in self.skip_array if i==epoch ][0]
                id_list, distance_mat = dynamic_time_warping(file_path_name)
                labels = aggomerativeClustering(self.file_path, distance_mat, self.dtw_clusters)
                labelled_weights = pd.read_csv(file_path_name)
                labelled_weights[cluster_id_verbose] = labels
                agglo_clust_mean = groupByClusters(labelled_weights)
                clust_reg_file_name = clus_reg_file_name_prefix + file_name_prefix + str(i).zfill(2) + file_name_suffix
                reg_data, new_headers = SimpLinearRegression(agglo_clust_mean, self.reg_train_steps,reg_prd_steps, self.file_path, clust_reg_file_name)

                #temporarilu store in new file , later to be stored in same file
                unclust_reg_weights_file = unclus_reg_file_name_prefix + file_name_prefix + str(i).zfill(2) + file_name_suffix
                reg_unclustered = uncluster_reg_weights(reg_data, new_headers, self.file_path, unclust_reg_weights_file)
                reg_unclustered.reset_index(drop=True, inplace=True)
                #later move REpoch to Epoch and add + r steps to eopchs
                #import pdb;pdb.set_trace()
                #import pdb;pdb.set_trace()
                df = pd.read_csv(file_path_name)
                custom_epoch = int(reg_unclustered.columns[-1].replace('Epoch:',''))

                df['Epoch:' + str(custom_epoch)] = reg_unclustered[reg_unclustered.columns[-1]]
                df.to_csv(file_path_name, index = False, header=True,float_format='%.3f')

                #feed returned weights , last col to network
                ripe_weights = reg_unclustered[reg_unclustered.columns[-1]]
                ripe_weights_np = np.zeros(shape=self.model.get_weights()[i].shape)
                for j in range(layer_weights_length):
                    ripe_weights_np[j] = ripe_weights[j*layer_weights_width:(j*layer_weights_width)+layer_weights_width]
                all_layer_weights[i] = ripe_weights_np
                #import pdb;pdb.set_trace()
                # Feed weights back to network
                self.model.set_weights(all_layer_weights)


    def on_epoch_end(self, epoch, logs={}):
        self.store_weights(epoch)
