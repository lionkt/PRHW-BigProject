class Config:
    SCALE=600
    MAX_SCALE=1200
    TEXT_PROPOSALS_WIDTH=16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO=0.8           # original: 0.5
    LINE_MIN_SCORE=0.93     # original: 0.9
    MAX_HORIZONTAL_GAP=25   # original: 50, but 25 is better
    TEXT_PROPOSALS_MIN_SCORE=0.6    # origianal: 0.7
    TEXT_PROPOSALS_NMS_THRESH=0.2   # origial: 0.2
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7




