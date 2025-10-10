## Why

The current PDF processing pipeline uses MinerU with vLLM for document layout analysis and text extraction. While this approach has served well, Docling's vision-language model (VLM) capabilities with the Gemma3 12B model offer several advantages:

1. **Better Document Understanding**: Gemma3 12B is a vision-language model that can understand document structure and content more holistically than traditional OCR approaches
2. **Improved Accuracy**: VLM models typically achieve higher accuracy on complex document layouts and mixed content types
3. **Enhanced Table Recognition**: Better handling of tables, figures, and structured content
4. **Reduced Infrastructure Complexity**: Single model approach instead of separate OCR + LLM pipeline
5. **Better Language Support**: Gemma3 supports multiple languages and specialized domains

## What Changes

This change proposal replaces the MinerU + vLLM PDF processing pipeline with Docling's VLM approach using the Gemma3 12B model:

- **BREAKING**: Replace MinerU service with Docling[vlm] integration (docling[vlm]>=2.0.0, transformers>=4.36.0)
- **BREAKING**: Update PDF processing pipeline stages to use Docling instead of MinerU
- **BREAKING**: Modify GPU service configuration for Gemma3 model requirements (~24GB VRAM)
- **BREAKING**: Update document processing pipeline to handle VLM output format (structured content extraction)
- **BREAKING**: Add DoclingVLMService class with same interface as MineruProcessor for compatibility
- Add comprehensive testing for the new VLM-based pipeline (unit, integration, performance, regression tests)
- Update configuration management for Docling-specific settings (model path, batch size, timeout, retry logic)
- Add feature flag system for gradual migration (0-100% traffic routing)
- Implement detailed monitoring and alerting for VLM processing metrics
- Update API documentation and error responses for VLM-specific behaviors

## Impact

- **Affected specs**: orchestration, services, config, gateway
- **Affected code**:
  - `src/Medical_KG_rev/services/mineru/` - Replace with Docling integration (`DoclingVLMService`)
  - `src/Medical_KG_rev/orchestration/stages/` - Update PDF processing stages (remove MinerU dependencies)
  - `src/Medical_KG_rev/config/` - Add Docling configuration (`DoclingVLMConfig` class)
  - `src/Medical_KG_rev/services/parsing/` - Extend Docling integration for PDF processing
  - `src/Medical_KG_rev/gateway/` - Update PDF processing endpoints (add VLM-specific error handling)
  - `src/Medical_KG_rev/services/gpu/` - Update GPU resource management for Gemma3 requirements
  - `src/Medical_KG_rev/observability/` - Add VLM-specific metrics and monitoring
- **New dependencies**:
  - `docling[vlm]>=2.0.0` - Vision-language model PDF processing
  - `transformers>=4.36.0` - Model loading and inference
  - `torch>=2.1.0` - PyTorch with CUDA support for GPU acceleration
  - `pillow>=10.0.0` - Image processing for PDF rendering
- **Migration**: Existing MinerU-processed documents remain accessible; new documents use Docling
- **Performance**: Expect 15-25% accuracy improvement but 20-30% longer processing times initially
- **Resource Requirements**: Gemma3 12B requires ~24GB VRAM vs ~8GB for current vLLM setup
