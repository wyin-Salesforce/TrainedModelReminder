import os

def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('starting model storing....')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('store succeed')
