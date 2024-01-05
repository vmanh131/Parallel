#include "dnnNetwork.h"

//void trainModel(int n_epoch, int batch_size)
int main()
{
    // data
    std::cout << "Loading fashion-mnist data...\n";
    MNIST dataset("./data/mnist/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

    std::cout << "Loading model...\n";

    float accuracy = 0.0;

     // 2. Host - CPU Network
    std::cout << "Test: Host - CPU Network" << std::endl;
    Network dnn1 = createNetwork_CPU();
    dnn1.load_parameters("./model/weights_cpu.bin");
    dnn1.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn1.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;
    std::cout << "==============================" << std::endl;

    // 3. Device - GPU Network
    Network dnn2 = createNetwork_GPU();
    dnn2.load_parameters("./model/weights_cpu.bin");
    dnn2.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn2.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;

    return 0;

    // //dnn.load_parameters("./model/weights.bin");
    // dnn.forward(dataset.test_data);
    // float accuracy = compute_accuracy(dnn.output(), dataset.test_labels);
    // std::cout << "test accuracy: " << accuracy << std::endl;
    

   // train & test
//     SGD opt(0.001, 5e-4, 0.9, true);

//     // train
//     for (int epoch = 0; epoch < n_epoch; epoch++)
//     {
//         shuffle_data(dataset.train_data, dataset.train_labels);
//         for (int start_idx = 0; start_idx < n_train; start_idx += batch_size)
//         {
//             int ith_batch = start_idx / batch_size;
//             Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
//                                                       std::min(batch_size, n_train - start_idx));
//             Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
//                                                             std::min(batch_size, n_train - start_idx));
//             Matrix target_batch = one_hot_encode(label_batch, 10);
//             if (false && ith_batch % 10 == 1)
//             {
//                 std::cout << ith_batch << "-th grad: " << std::endl;
//                 dnn.check_gradient(x_batch, target_batch, 10);
//             }
//             dnn.forward(x_batch);
//             dnn.backward(x_batch, target_batch);
//             // display
//             if (ith_batch % 50 == 0)
//             {
//                 std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss()
//                           << std::endl;
//             }
//             // optimize
//             dnn.update(opt);
//         }

//         // test
//         dnn.forward(dataset.test_data);
//         float acc = compute_accuracy(dnn.output(), dataset.test_labels);
//         std::cout << std::endl;
//         std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
//         std::cout << std::endl;
//     }
}

// int main(int argc, char *argv[])
// {

//     trainModel(5, 128);

//     return 0;
// }