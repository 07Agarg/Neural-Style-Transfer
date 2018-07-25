# Neural-Style-Transfer

To obtain a representation of the STYLE of an input image : using a feature space to obtain texture information of an input image and not the global imformation. Feature space is built by obtaining filter responses of each layer of convolution neural networks. Correlations between different filter responses over the spatial content of an image. 

To visualise the information at different processing stages in the CNN by reconstructing the input image from only knowing the network's responses in a particular layer. Reconstructing input image form layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1) from lower layers is almost perfect because it contains detailed pixel information of the image while higher layers in the network captures the high level content in terms of objects and their arrangements in the input image but do not constrain the exact pixel values of the reconstructions. 

Therefore, feature responses in higher layers of networks are considered as Content Representations. 

Content Reconstructions. We can visualise the information at different processing stages in the CNN by reconstructing the input image from only knowing
the network’s responses in a particular layer. We reconstruct the input image from from layers ‘conv1 1’ (a), ‘conv2 1’ (b), ‘conv3 1’ (c), ‘conv4 1’ (d) and ‘conv5 1’ (e) of the original VGG-Network. We find that reconstruction from lower layers is almost perfect (a,b,c). In higher layers of the network, detailed pixel information is lost while the high-level content of the image is preserved (d,e). 

Style Reconstructions. On top of the original CNN representations we built a new feature space that captures the style of an input image. The style representation computes correlations between the different features in different layers of the CNN. We reconstruct the style of the input image from style representations built on different subsets of CNN layers ( ‘conv1 1’ (a), ‘conv1 1’ and ‘conv2 1’ (b), ‘conv1 1’, ‘conv2 1’ and ‘conv3 1’ (c), ‘conv1 1’, ‘conv2 1’, ‘conv3 1’ and ‘conv4 1’ (d), ‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’ and ‘conv5 1’ (e)). This creates images that match the style of a given image on an increasing scale while discarding information of the global arrangement of the scene.

Reconstructions from the style features produce texturised versions of the input image that capture its general appearance in terms of colour and localised structures.
The style representation is a multi-scale representation that includes multiple layers of the neural network. Style is defined as correlations between activations across channels. For this compute style matrix. (G) . values in G matrix will be large when features are highly correlated and small values when uncorrelated. 
