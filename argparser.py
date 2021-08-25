import argparse
import tasks


def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21
    if opts.dataset == 'ade':
        opts.num_classes = 150
    if opts.dataset == 'cityscapes':
        opts.num_classes = 20
    if opts.dataset == 'gta':
        opts.num_classes = 20
		
    if not opts.visualize:
        opts.sample_num = 0

    if opts.where_to_sim == 'GPU_server':
        opts.net_pytorch = False

    if opts.method is not None:
        if opts.method == 'FT':
            pass
        if opts.method == 'LWF':
            opts.loss_kd = 100
        if opts.method == 'CIL':
            opts.loss_CIL == 1
        if opts.method == 'LWF-MC':
            opts.icarl = True
            opts.icarl_importance = 10
        if opts.method == 'ILT':
            opts.loss_kd = 100
            opts.loss_de = 100
        if opts.method == 'EWC':
            opts.regularizer = "ewc"
            opts.reg_importance = 1000
        if opts.method == 'RW':
            opts.regularizer = "rw"
            opts.reg_importance = 1000
        if opts.method == 'PI':
            opts.regularizer = "pi"
            opts.reg_importance = 1000
        if opts.method == 'MiB':
            #opts.loss_kd = 50
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = True
        if opts.method == 'IL-UDA':
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = True        
        if opts.method == 'UDA' or opts.method == 'IL-UDA':
            pass
            #opts.uda_lmsq = 4
            #opts.lIWmsq = 0.2
            #if opts.target_dataset is None:
            #raise ValueError('[!] PARAMETERS ERROR: Select a target dataset to use domain adaptation')
                    
    opts.no_overlap = not opts.overlap
    opts.no_cross_val = not opts.cross_val

    opts.uda_loss = (opts.uda_lce + opts.uda_lsce + opts.uda_lmsq + opts.uda_lIWsce + opts.uda_lIWmsq + opts.uda_lkd + opts.uda_lfmsq + opts.uda_lfIWmsq) > 0.
    opts.uda_target = opts.target_dataset is not None
    opts.uda_validate_target = opts.uda_target and not opts.uda_validate_source
    
    opts.low_loss = (opts.low_lce + opts.low_lsce + opts.low_lmsq + opts.low_lIWsce + opts.low_lIWmsq) > 0.
    opts.multi = opts.multiDeeplab and opts.low_loss

    opts.uda = opts.uda_target and (opts.uda_loss or opts.multi)

    if opts.batch_size < 1 or (opts.target_batch_size < 1 and opts.uda):
        raise ValueError('[!] PARAMETERS ERROR: Batches sizes must be greater that 1')
        
    opts.total_batch_size = opts.batch_size + opts.target_batch_size

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # NB: on CPU not feasible because of inplace_ABN functions.
    # on GPU_windows need to remove apex since not supported
    # on GPU_server code as it has been downloaded
    parser.add_argument('--where_to_sim', type=str, choices=['GPU_windows', 'GPU_server', 'CPU', 'CPU_windows'], default='GPU_server')
    parser.add_argument("--net_pytorch", action='store_false', default=True,
                        help='whether to use default resnet from pytorch or to use the network as in MiB (default: True)')
    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help='number of workers (default: 1)')

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./dataset',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'ade', 'cityscapes', 'gta'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None), set by method modify_command_options()")
    parser.add_argument("--filter_unused_labels", action='store_true', default=False,
                        help="filter out labels that are not part of the task, set them to background (class: 0)")						

    # Method Options
    # BE CAREFUL USING THIS, THEY WILL OVERRIDE ALL THE OTHER PARAMETERS.
    # This argument serves to use default parameters for the methods defined in function: modify_command_options()
    parser.add_argument("--method", type=str, default=None,
                        choices=['FT', 'LWF', 'LWF-MC', 'ILT', 'EWC', 'RW', 'PI', 'MiB', 'CIL', 'UDA', 'IL-UDA'],
                        help="The method you want to use. BE CAREFUL USING THIS, IT MAY OVERRIDE OTHER PARAMETERS.")

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")
    parser.add_argument("--fix_bn", action='store_true', default=False,
                        help='fix batch normalization during training (default: False)')

    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")

    parser.add_argument("--lr", type=float, default=0.007,
                        help="learning rate (default: 0.007)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step'], help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--bce", default=False, action='store_true',
                        help="Whether to use BCE or not (default: no)")

    # whether to consider clustering on feature spaces as loss
    parser.add_argument("--loss_fc", type=float, default=0.,  # Features Clustering
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable features clustering loss")
    parser.add_argument("--lfc_L2normalized", action='store_true', default=False,
                        help="enable features clustering loss L2 normalized (default False)")
    parser.add_argument("--lfc_nobgr", action='store_true', default=False,
                        help="enable features clustering loss without background (default False)")
    parser.add_argument("--lfc_orth_sep", action='store_true', default=False,
                        help="Orthogonal separation loss applied on the current prototypes only")
    parser.add_argument("--lfc_orth_maxonly", action='store_true', default=False,
                        help="Orthogonal separation loss, only the maximum value is considered")
    parser.add_argument("--lfc_sep_clust", type=float, default=0.,  # Separation of Clusters
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable separation between clusters loss")
    parser.add_argument("--lfc_sep_clust_ison_proto", action='store_true', default=False,
                        help="enable separation clustering loss on prototypes (default False)")
    # whether to consider Soft Nearest Neighbor Loss (SNNL) as loss at features space
    parser.add_argument("--loss_SNNL", type=float, default=0.,  # SNNL
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable SNNL at feature level")
    parser.add_argument("--loss_featspars", type=float, default=0.,  # features sparsification
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable features sparsification loss")
    parser.add_argument("--lfs_normalization", type=str, default='max_foreachfeature',
                        choices=['L1', 'L2', 'max_foreachfeature', 'max_maskedforclass', 'max_overall', 'softmax'],
                        help="The method you want to use to normalize lfs")
    parser.add_argument("--lfs_shrinkingfn", type=str, default='squared',
                        choices=['squared', 'power3', 'exponential'],
                        help="The method you want to use to shrink the lfs")
    parser.add_argument("--lfs_loss_fn_touse", type=str, default='ratio',
                        choices=['ratio', 'max_minus_ratio', 'lasso', 'entropy'],
                        help="The loss function you want to use for the lfs")
    parser.add_argument("--loss_bgruncertainty", type=float, default=0.,
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable background uncertainty loss")
    parser.add_argument("--lbu_inverse", action='store_true', default=False,
                        help="enable inverse on lbu loss")
    parser.add_argument("--lbu_mean", action='store_true', default=False,
                        help="enable lbu_mean on lbu loss")
    parser.add_argument("--loss_CIL", type=float, default=0.,
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable loss of CIL paper")
    parser.add_argument("--feat_dim", type=float, default=2048,
                        help="Dimensionality of the features space (default: 2048 as in Resnet-101)")

    # Validation Options
    parser.add_argument("--val_on_trainset", action='store_true', default=False,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--cross_val", action='store_true', default=False,
                        help="If validate on training or on validation (default: Train)")
    parser.add_argument("--crop_val", action='store_false', default=True,
                        help='do crop for validation (default: True)')

    # Logging Options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=0,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug",  action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize",  action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_print_interval", type=int, default=0,
                        help="print interval of loss (default: 0, no print)")
    parser.add_argument("--val_interval", type=int, default=15,
                        help="epoch interval for eval (default: 15)")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")

    # Model Options
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Wheather to use pretrained or not (def: True)')

    # Test and Checkpoint options
    parser.add_argument("--test",  action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--round", type=int, default=0,
                        help="Indicate that training is part of a round approach")
    parser.add_argument("--continue_with_round",  default=False, action='store_true',
                        help="for rounds, if true continue previous round (defaul: False)") 
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--ckpt_resume_training",  action='store_true', default=False,
                        help="Whether to use the checkpoint to resume the training or for a new session (def: do not resume)")

    # Parameters for Knowledge Distillation of ILTSS (https://arxiv.org/abs/1907.13372)
    parser.add_argument("--freeze", action='store_true', default=False,
                        help="Use this to freeze the feature extractor in incremental steps")
    parser.add_argument("--loss_de", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable distillation on Encoder (L2)")
    parser.add_argument("--loss_de_maskedold", default=False, action='store_true',
                        help="If enabled, loss_de is masked to consider only old classes features (default: False)")
    parser.add_argument("--loss_de_prototypes", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable loss_de with prototypes (idea 1b)")
    parser.add_argument("--loss_de_prototypes_sumafter", action='store_true', default=False,
                        help="....TODO....")
    parser.add_argument("--loss_de_cosine", action='store_true', default=False,
                        help="....TODO....")
    parser.add_argument("--loss_kd", type=float, default=0.,  # Distillation on Output
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable Knowledge Distillation (Soft-CrossEntropy)")

    # Parameters for EWC, RW, and SI (from Riemannian Walks https://arxiv.org/abs/1801.10112)
    parser.add_argument("--regularizer", default=None, type=str, choices=['ewc', 'rw', 'pi'],
                        help="regularizer you want to use. Default is None")
    parser.add_argument("--reg_importance", type=float, default=1.,
                        help="set this par to a value greater than 0 to enable regularization")
    parser.add_argument("--reg_alpha", type=float, default=0.9,
                        help="Hyperparameter for RW and EWC that controls the update of Fisher Matrix")
    parser.add_argument("--reg_no_normalize", action='store_true', default=False,
                        help="If EWC, RW, PI must be normalized or not")
    parser.add_argument("--reg_iterations", type=int, default=10,
                        help="If RW, the number of iterations after each the update of the score is done")

    # Arguments for ICaRL (from https://arxiv.org/abs/1611.07725)
    parser.add_argument("--icarl", default=False, action='store_true',
                        help="If enable ICaRL or not (def is not)")
    parser.add_argument("--icarl_importance",  type=float, default=1.,
                        help="the regularization importance in ICaRL (def is 1.)")
    parser.add_argument("--icarl_disjoint", action='store_true', default=False,
                        help="Which version of icarl is to use (def: combined)")
    parser.add_argument("--icarl_bkg", action='store_true', default=False,
                        help="If use background from GT (def: No)")

    # METHODS
    parser.add_argument("--init_balanced", default=False, action='store_true',
                        help="Enable Background-based initialization for new classes")
    parser.add_argument("--unkd", default=False, action='store_true',
                        help="Enable Unbiased Knowledge Distillation instead of Knowledge Distillation")
    parser.add_argument("--alpha", default=1., type=float,
                        help="The parameter to hard-ify the soft-labels. Def is 1.")
    parser.add_argument("--unce", default=False, action='store_true',
                        help="Enable Unbiased Cross Entropy instead of CrossEntropy")

    # Incremental parameters
    parser.add_argument("--task", type=str, default="19-1", choices=tasks.get_task_list(),
                        help="Task to be executed (default: 19-1)")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    # Consider the dataset as done in
    # http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.pdf
    # and https://arxiv.org/pdf/1911.03462.pdf : same as disjoint scenario (default) but with label of old classes in
    # new images, if present.
    parser.add_argument("--no_mask", action='store_true', default=False,
                        help="Use this to not mask the old classes in new training set, i.e. use labels of old classes"
                             " in new training set (if present)")
    parser.add_argument("--overlap", action='store_true', default=False,
                        help="Use this to not use the new classes in the old training set")
    parser.add_argument("--step_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")
    parser.add_argument('--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0')


    #UDA parameters
    parser.add_argument("--target_dataset", type=str, default=None,
                        choices=['voc', 'ade', 'cityscapes', 'gta'], help='Name of target dataset, also validate on target dataset and not source')
    parser.add_argument("--target_batch_size", type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument("--uda_lce", type=float, default=0.,
                        help="Add a hard cross entropy loss in the target dataset loss (set >0 to activate)")
    parser.add_argument("--uda_lsce", type=float, default=0.,
                        help="Add a soft entropy in the target dataset loss (set >0 to activate)")
    parser.add_argument("--uda_lmsq", type=float, default=0.,
                        help="Add a max square loss loss in the target dataset loss (set >0 to activate)")
    parser.add_argument("--uda_lIWsce",  type=float, default=0.,
                        help="the ratio parameter for a Image-wise Class Balanced soft entropy loss (Must be > 0), (default: 0)")
    parser.add_argument("--uda_lIWmsq",  type=float, default=0.,
                        help="the ratio parameter for a Image-wise Class Balanced max square loss (Must be > 0), (default: 0)")
    parser.add_argument("--uda_lkd", type=float, default=0.,
                        help="Add a knowledge distillation loss in the target dataset loss (set >0 to activate)")                        
    parser.add_argument("--uda_lkd_alpha", type=float, default=1.,
                        help="knowledge distillation temperature")
    parser.add_argument("--uda_validate_source", default=False, action='store_true',
                        help='force validation on source even if uda active')
    parser.add_argument("--uda_unkd", default=False, action='store_true',
                        help='If selected normalize Image-wise class balance')
                        
    # multi-level self-produced  guidance for uda loss
    parser.add_argument("--multiDeeplab", default=False, action='store_true',
                        help='use deeplab low level features for multi-levels predictions')
    parser.add_argument("--multi_delta", type=float, default=0.,
                        help='delta value for the ensemble of low level and high level predictions') 
    parser.add_argument("--low_lce", type=float, default=0.,
                        help="Add a low level hard cross entropy loss in the target dataset loss (set >0 to activate)")
    parser.add_argument("--low_lsce", type=float, default=0.,
                        help="Add a low level soft entropy in the target dataset loss (set >0 to activate)")
    parser.add_argument("--low_lmsq", type=float, default=0.,
                        help="Add a low level max square loss loss in the target dataset loss (set >0 to activate)")
    parser.add_argument("--low_lIWsce",  type=float, default=0.,
                        help="the ratio parameter for a Image-wise Class Balanced low level soft entropy loss (Must be > 0), (default: 0)")
    parser.add_argument("--low_lIWmsq",  type=float, default=0.,
                        help="the ratio parameter for a Image-wise Class Balanced low level max square loss (Must be > 0), (default: 0)")

    #source Image-wise losses
    parser.add_argument("--src_lIWce",  type=float, default=0.,
                        help="the ratio parameter for a Image-wise Class Balanced low level cross entropy loss (Must be > 0), (default: 0)")
    parser.add_argument("--src_lIWkd",  type=float, default=0.,
                        help="the ratio parameter for a Image-wise Class Balanced low knowldge distillation loss (Must be > 0), (default: 0)")

    # weight sampler
    parser.add_argument("--weight_samples", type=float, default=0.,
                        help='If (> 0) selected sample are weighted in the source to improve rare classes')

    # test number to keep track of each test                    
    parser.add_argument("--backup",  type=int, default=0.,
                        help="Paramenter to number each test and run multiple test while non overwriting previous saved model")     

    # test with msq modifications
    parser.add_argument("--group_msq", default=False, action='store_true',
                        help='If selected group all step unrelated labels to the background like MiB, (default: False)')      
    parser.add_argument("--new_msq", default=False, action='store_true',
                        help='If selected always set adaptation factor for old labels to 1, (default: False)')   
    parser.add_argument("--fix_old_cl",  type=int, default=0.,
                        help="If > 0 train with set fixed old classes, (default: 0)")    

    # feature level knowledge distillation and domain adaptation
    parser.add_argument("--lkd_features", default=False, action='store_true',
                        help='If selected use knowledge distillation at feature level')
    parser.add_argument("--lkd_features_target", default=False, action='store_true',
                        help='If selected use knowledge distillation at feature level on the target domain')
    parser.add_argument("--uda_lfmsq", type=float, default=0.,
                        help="Add a low level max square loss loss in the target dataset loss (set >0 to activate)")
    parser.add_argument("--uda_lfIWmsq",  type=float, default=0.,
                        help="the ratio parameter for a Image-wise Class Balanced low level max square loss (Must be > 0), (default: 0)")                          
    return parser
