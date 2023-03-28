dataset_info = dict(
    dataset_name='MidvDataset',
    paper_info=dict(
        author='V.V. Arlazarov, K. Bulatov, T. Chernov, V.L. Arlazarov',
        title='MIDV-500: a dataset for identity document analysis and recognition on mobile devices in video stream',
        container='Computer Optics 2019',
        year='2019',
        homepage='https://github.com/fcakyon/midv500',
    ),
    keypoint_info={
        0: dict(name='top_left', id=0, color=[0, 255, 0], type='upper', swap='top_right'),
        1: dict(name='top_right', id=1, color=[0, 255, 0], type='upper', swap='top_left'),
        2: dict(name='bottom_right', id=2, color=[0, 255, 0], type='lower', swap='bottom_left'),
        3: dict(name='bottom_left', id=3, color=[0, 255, 0], type='lower', swap='bottom_right')
    },
    skeleton_info={
        0: dict(link=('top_left', 'top_right'), id=0, color=[0, 255, 0]),
        1: dict(link=('top_right', 'bottom_right'), id=1, color=[0, 255, 0]),
        2: dict(link=('bottom_right', 'bottom_left'), id=2, color=[0, 255, 0]),
        3: dict(link=('bottom_left', 'top_left'), id=3, color=[0, 255, 0])
    },
    joint_weights=[1., 1., 1., 1.],
    sigmas=[0.25, 0.25, 0.25, 0.25]
)
