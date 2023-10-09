
Returning nan 

Removing skip connections doesnt work
Introducing dropout doesnt work
    - Activation function?
    - log(0) anywhere?


With just
2 conv2d
1 dense layer
still NaN

Since we are not applying a softmax at the end, from_logits=True
Check input data?


Try with just 1 conv2d layer and 1 dense layer
-   Still NaN

Try with batch normalizing
-   Still Nan

Try with just softmaxing the inputs
-   Works!?

Dense + Softmax
-   Does not work... Dense layer is the issue?
-   Perhaps using flatten and a real dense layer instead of 1x1 conv?

Replace Dense + softmax with 1x1 conv using sigmoid activation
- NaN??
- Conv2D(name='classification', filters=classes, kernel_size=(1,1), activation='sigmoid', padding='same')(input)


💚 Use sigmoid Activation() layer
- Works on the 2d data! ?? should be identical to 1x1 conv tho?

🔴 Try 1 encoder layer with sigmoid laye

🔴 Try 1 conv2d with sigmoid layer

💚 Try 1 conv2d with sigmoid layer With binary cross entropy

💚 <100 epochs> Try 1 conv2d with sigmoid layer With binary cross entropy

Issue was categorical crossentropy was expecting logits for both classes.
I was only returning one prediction per class.
Binary crossentropy loss would make this  model compatible and trianable.


`*~,.,~*`*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*`
`Optimize` the `Learning!`
`*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*`

- \>\> Class weights
- Learning rate scheduler
- Image augmentation

`*~,.,~*`*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*`
`Different` Models
`*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*``*~,.,~*`

- Attention UNet