-- Face Re-ID Database Schema

-- Bảng lịch sử điểm danh
CREATE TABLE IF NOT EXISTS attendance_log (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    similarity  REAL,
    image_path  VARCHAR(500),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bảng log người lạ (Unknown)
CREATE TABLE IF NOT EXISTS unknown_log (
    id          SERIAL PRIMARY KEY,
    image_path  VARCHAR(500),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    note        TEXT
);

-- Bảng settings ứng dụng
CREATE TABLE IF NOT EXISTS app_settings (
    key   VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL
);

-- Default settings
INSERT INTO app_settings (key, value) VALUES
    ('det_weight',           './weights/det_10g.onnx'),
    ('rec_weight',           './weights/w600k_mbf.onnx'),
    ('confidence_thresh',    '0.5'),
    ('similarity_thresh',    '0.4'),
    ('unknown_debounce_sec', '5'),
    ('known_debounce_min',   '1')
ON CONFLICT (key) DO NOTHING;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_attendance_name ON attendance_log (name);
CREATE INDEX IF NOT EXISTS idx_attendance_time ON attendance_log (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_unknown_time    ON unknown_log   (detected_at DESC);