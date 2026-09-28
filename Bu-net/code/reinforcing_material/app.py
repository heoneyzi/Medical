"""Local viewer for user-exported segmentation results; no model inference."""
from pathlib import Path
import os

import streamlit as st
from PIL import UnidentifiedImageError

from mask_index import SERIES, discover_cases, load_case_images


def main():
    st.set_page_config(page_title='Brain tumor segmentation | 보강재', layout='wide')
    st.title('Brain tumor segmentation experiments')
    st.caption('deep daiv. 2024 · 보강재 · Jiheon Kang, Boyoung Kwon, Jaeryeong Hwang')
    st.write('Compare a ground-truth mask with three sets of exported model results.')
    folder = st.sidebar.text_input('Local image directory', value=os.environ.get('DEMO_DATA_DIR', 'test_data'))
    directory = Path(folder).expanduser()
    st.sidebar.caption('Match Model 1, Model 2 and Model 3 to your own experiment log.')

    try:
        complete, available = discover_cases(directory)
    except OSError as error:
        st.error(f'Cannot read the image directory: {error}')
        return

    all_ids = set.union(*available.values())
    incomplete = all_ids.difference(complete)
    if incomplete:
        st.warning(f'{len(incomplete)} case(s) have missing comparison files and are not shown.')

    if not complete:
        st.info('Add your own exported PNG masks to open the comparison viewer. No MRI data or model predictions are bundled.')
        st.code('\n'.join(f'{prefix}_1.png' for prefix, _ in SERIES), language='text')
        st.caption('Use the same positive integer for the four files belonging to each case. Any number of complete cases is supported.')
        return

    case_id = st.selectbox('Case', complete)
    st.caption(f'{len(complete)} complete case(s) found. These images are supplied by the local user.')
    try:
        images = load_case_images(directory, case_id)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        st.error(f'Cannot open this case: {error}')
        return

    for column, (label, image) in zip(st.columns(len(images)), images):
        column.image(image, caption=label)
    st.caption('This viewer displays existing images. It does not train a model, run inference or measure segmentation quality.')


if __name__ == '__main__':
    main()
