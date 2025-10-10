-- Migration: add_docling_config
-- Purpose: Introduce Docling VLM configuration storage and register the new backend feature flag.

BEGIN;

CREATE TABLE IF NOT EXISTS docling_vlm_config (
    id SERIAL PRIMARY KEY,
    model_version TEXT NOT NULL,
    model_path TEXT NOT NULL,
    batch_size INTEGER NOT NULL DEFAULT 8,
    timeout_seconds INTEGER NOT NULL DEFAULT 300,
    retry_attempts INTEGER NOT NULL DEFAULT 3,
    gpu_memory_fraction NUMERIC(5,4) NOT NULL DEFAULT 0.9500,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docling_vlm_config_model_version
    ON docling_vlm_config (model_version);

CREATE INDEX IF NOT EXISTS idx_docling_vlm_config_enabled
    ON docling_vlm_config (enabled);

-- Ensure a default Docling configuration row exists for the Gemma3 12B model
INSERT INTO docling_vlm_config (model_version, model_path, batch_size, timeout_seconds, retry_attempts, gpu_memory_fraction)
VALUES ('gemma3-12b-v1', '/models/gemma3-12b', 8, 300, 3, 0.9500)
ON CONFLICT (model_version) DO UPDATE
    SET model_path = EXCLUDED.model_path,
        batch_size = EXCLUDED.batch_size,
        timeout_seconds = EXCLUDED.timeout_seconds,
        retry_attempts = EXCLUDED.retry_attempts,
        gpu_memory_fraction = EXCLUDED.gpu_memory_fraction,
        enabled = TRUE,
        updated_at = NOW();

-- Register Docling feature flag if a feature_flags table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'feature_flags') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'feature_flags' AND column_name = 'name') THEN
            EXECUTE $$
                INSERT INTO feature_flags (name, is_enabled, description)
                VALUES ('pdf_processing_backend_docling', TRUE, 'Enable Docling Gemma3 VLM processing pipeline')
                ON CONFLICT (name) DO UPDATE
                SET is_enabled = EXCLUDED.is_enabled,
                    description = EXCLUDED.description;
            $$;
        END IF;
    END IF;
END$$;

-- Optionally register Docling backend in pdf_processing_backends lookup table when available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'pdf_processing_backends') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'pdf_processing_backends' AND column_name = 'name') THEN
            EXECUTE $$
                INSERT INTO pdf_processing_backends (name, is_active, notes)
                VALUES ('docling_vlm', TRUE, 'Docling Gemma3 12B VLM pipeline')
                ON CONFLICT (name) DO UPDATE
                SET is_active = EXCLUDED.is_active,
                    notes = EXCLUDED.notes;
            $$;
        END IF;
    END IF;
END$$;

COMMIT;
