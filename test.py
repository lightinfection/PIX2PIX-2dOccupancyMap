import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
mask_values = data_loader.dataset.mask_values if opt.mask_output else None
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
model = create_model(opt)
if opt.data_type == 16:
    model.half()
elif opt.data_type == 8:
    model.type(torch.uint8)
        
if opt.verbose:
    print(model)
    
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
    minibatch = 1     
    generated = model.inference(data['label'], data['image'])
    
    if not opt.mask_output:
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc, normalize=(not opt.no_norm_input), stats=(not opt.define_norm))),
                            ('synthesized_image', util.tensor2im(generated.data[0], normalize=(not opt.no_norm_input), default_stats=(not opt.define_norm)))])
    else:
        visuals = OrderedDict([('input_label', util.masks2im(data['label'][0], mask_values=mask_values)),
                            ('synthesized_image', util.masks2im(generated.data[0], mask_values=mask_values))])
       
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
