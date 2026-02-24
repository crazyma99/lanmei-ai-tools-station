import React, { useState } from 'react';
import {
  Card,
  Descriptions,
  Upload,
  Button,
  Space,
  Select,
  Slider,
  Message,
  Typography,
  Checkbox,
  Input,
  Modal,
} from '@arco-design/web-react';
import type { RequestOptions } from '@arco-design/web-react/es/Upload';
import axios from 'axios';

const { Text, Paragraph } = Typography;

interface ExifUploadResponse {
  id: string;
  filename: string;
  thumbnail_url: string;
  exif: Record<string, unknown>;
  aigc: boolean;
  aigc_detail: Record<string, unknown>;
  width: number | null;
  height: number | null;
  format: string | null;
}

type PresetKey = 'none' | 'sony_a7m4' | 'fuji_xt5' | 'hasselblad_x2d' | 'custom';

const ExifToolPage: React.FC = () => {
  const [items, setItems] = useState<ExifUploadResponse[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [convertToJpg, setConvertToJpg] = useState(false);
  const [addNoise, setAddNoise] = useState(false);
  const [noiseIntensity, setNoiseIntensity] = useState(10);
  const [clearAigc, setClearAigc] = useState(true);
  const [deepClean, setDeepClean] = useState(false);
  const [preset, setPreset] = useState<PresetKey>('none');
  const [customJson, setCustomJson] = useState('');
  const [customJsonError, setCustomJsonError] = useState<string | null>(null);
  const [processedIds, setProcessedIds] = useState<string[]>([]);
  const [processingAll, setProcessingAll] = useState(false);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewItem, setPreviewItem] = useState<ExifUploadResponse | null>(null);
  const [resultPreviewItem, setResultPreviewItem] =
    useState<ExifUploadResponse | null>(null);
  const [resultPreviewVisible, setResultPreviewVisible] = useState(false);

  const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || '',
  });

  const activeItem =
    items.find((item) => item.id === activeId) || items[0] || null;

  const handleUpload = async (options: RequestOptions) => {
    const { file } = options;
    if (!file) return;
    const form = new FormData();
    form.append('file', file as File);
    try {
      const res = await api.post<ExifUploadResponse>('/api/exif/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setItems((prev) => [...prev, res.data]);
      setActiveId(res.data.id);
      Message.success('上传成功并解析完成');
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } } };
      Message.error(err?.response?.data?.detail || '上传失败');
    }
    if (options.onSuccess) {
      options.onSuccess();
    }
  };

  const buildPayloadForId = (id: string) => {
    let action: 'clear' | 'deep_clean' | 'import_preset' | 'import_custom' = 'clear';
    const payload: Record<string, unknown> = {
      id,
      convert_to_jpg: convertToJpg,
      add_noise: addNoise,
      noise_intensity: noiseIntensity,
      clear_aigc: clearAigc,
      deep_clean: deepClean,
    };

    if (preset === 'custom') {
      if (!customJson.trim()) {
        Message.warning('请先填写自定义 EXIF JSON');
        return null;
      }
      try {
        const parsed = JSON.parse(customJson);
        action = 'import_custom';
        (payload as { custom_data?: unknown }).custom_data = parsed;
        setCustomJsonError(null);
      } catch {
        setCustomJsonError('JSON 格式不正确，请检查括号和引号');
        Message.error('自定义 JSON 解析失败');
        return null;
      }
    } else if (preset !== 'none') {
      action = 'import_preset';
      if (preset === 'sony_a7m4') {
        payload.preset = 'sony_a7m4';
      } else if (preset === 'fuji_xt5') {
        payload.preset = 'fuji_xt5';
      } else if (preset === 'hasselblad_x2d') {
        payload.preset = 'hasselblad_x2d';
      }
    } else if (deepClean) {
      action = 'deep_clean';
    } else {
      action = 'clear';
    }

    payload.action = action;
    return payload;
  };

  const applyProcessToId = async (id: string) => {
    const payload = buildPayloadForId(id);
    if (!payload) return;
    const res = await api.post('/api/exif/process', payload);
    const data = res.data as {
      exif: Record<string, unknown>;
      aigc: boolean;
      aigc_detail: Record<string, unknown>;
      width?: number | null;
      height?: number | null;
      format?: string | null;
    };
    setItems((prev) =>
      prev.map((item) =>
        item.id === id
          ? {
              ...item,
              exif: data.exif,
              aigc: data.aigc,
              aigc_detail: data.aigc_detail,
              width: data.width ?? item.width,
              height: data.height ?? item.height,
              format: data.format ?? item.format,
            }
          : item,
      ),
    );
    setProcessedIds((prev) => (prev.includes(id) ? prev : [...prev, id]));
  };

  const handleProcess = async () => {
    if (!activeItem) {
      Message.warning('请先上传图片');
      return;
    }
    setProcessing(true);
    try {
      await applyProcessToId(activeItem.id);
      Message.success('处理完成');
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } } };
      Message.error(err?.response?.data?.detail || '处理失败');
    } finally {
      setProcessing(false);
    }
  };

  const handleProcessAll = async () => {
    if (!items.length) {
      Message.warning('请先上传图片');
      return;
    }
    setProcessingAll(true);
    try {
      for (const item of items) {
        try {
          await applyProcessToId(item.id);
        } catch (e) {
          const err = e as { response?: { data?: { detail?: string } } };
          Message.error(
            err?.response?.data?.detail || `图片 ${item.filename} 处理失败`,
          );
        }
      }
      Message.success('批量处理完成');
    } finally {
      setProcessingAll(false);
    }
  };

  const handleDownload = () => {
    if (!activeItem) return;
    const base = import.meta.env.VITE_API_BASE_URL || '';
    window.open(`${base}/api/exif/download/${activeItem.id}`, '_blank');
  };

  const handleBatchDownload = async () => {
    if (!processedIds.length) {
      Message.warning('暂无可下载的处理结果');
      return;
    }
    try {
      const res = await api.post(
        '/api/exif/download_batch',
        { ids: processedIds },
        { responseType: 'blob' },
      );
      const blob = new Blob([res.data as BlobPart], { type: 'application/zip' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'exif_batch_results.zip';
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } } };
      Message.error(err?.response?.data?.detail || '批量下载失败');
    }
  };

  const handleClearResults = () => {
    setProcessedIds([]);
  };

  const handleClearUpload = () => {
    setActiveId(null);
    setItems([]);
  };

  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';

  const handleDeepCleanToggle = (v: boolean) => {
    setDeepClean(v);
    setClearAigc(v);
  };

  const handleDownloadById = (id: string) => {
    const base = import.meta.env.VITE_API_BASE_URL || '';
    window.open(`${base}/api/exif/download/${id}`, '_blank');
  };

  const flattenExif = (exif: Record<string, unknown>) => {
    const flat: Record<string, unknown> = {};
    Object.entries(exif).forEach(([key, value]) => {
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        Object.entries(value as Record<string, unknown>).forEach(
          ([innerKey, innerValue]) => {
            flat[innerKey] = innerValue;
          },
        );
      } else {
        flat[key] = value;
      }
    });
    return flat;
  };

  const renderAllMetadata = (exif: Record<string, unknown>) => {
    const flat = flattenExif(exif);
    const entries = Object.entries(flat);
    if (!entries.length) {
      return (
        <Text type="secondary">
          无元数据
        </Text>
      );
    }
    return (
      <div
        style={{
          maxHeight: 360,
          overflow: 'auto',
        }}
      >
        {entries.map(([key, value]) => {
          let display = '';
          if (value == null) {
            display = '-';
          } else if (Array.isArray(value)) {
            display = value.map((v) => String(v)).join(', ');
          } else {
            display = String(value);
          }
          return (
            <div
              key={key}
              style={{
                fontSize: 12,
                color: '#4e5969',
                lineHeight: 1.6,
              }}
            >
              <span
                style={{
                  fontWeight: 500,
                  marginRight: 4,
                }}
              >
                {key}:
              </span>
              {display}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={16}>
      <Card title="处理配置">
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          <Space
            wrap
            size={20}
            style={{ width: '100%', justifyContent: 'center' }}
          >
            <Checkbox
              checked={convertToJpg}
              onChange={(v) => setConvertToJpg(v)}
            >
              转为 JPG 格式
            </Checkbox>
            <Checkbox
              checked={deepClean && clearAigc}
              onChange={handleDeepCleanToggle}
            >
              深度抹除并清除标签（AIGC）
            </Checkbox>
            <Space align="center">
              <Checkbox
                checked={addNoise}
                onChange={(v) => setAddNoise(v)}
              >
                启用增加颗粒
              </Checkbox>
              {addNoise && (
                <Slider
                  style={{ width: 180 }}
                  min={0}
                  max={100}
                  value={noiseIntensity}
                  onChange={(v) => {
                    const value = v as number;
                    setNoiseIntensity(value);
                  }}
                />
              )}
            </Space>
          </Space>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 12,
              justifyContent: 'center',
              width: '100%',
            }}
          >
            <Select
              style={{ width: 220 }}
              value={preset}
              onChange={(v) => setPreset(v as PresetKey)}
            >
              <Select.Option value="none">无预设（仅清洗 / 变换）</Select.Option>
              <Select.Option value="sony_a7m4">Sony A7M4 预设</Select.Option>
              <Select.Option value="fuji_xt5">Fuji X-T5 预设</Select.Option>
              <Select.Option value="hasselblad_x2d">
                Hasselblad X2D 预设
              </Select.Option>
              <Select.Option value="custom">自定义 JSON</Select.Option>
            </Select>
            <Button
              type="primary"
              loading={processing}
              onClick={handleProcess}
            >
              开始处理
            </Button>
            <Button
              loading={processingAll}
              onClick={handleProcessAll}
            >
              批量处理全部图片
            </Button>
          </div>
        </Space>
        {preset === 'custom' && (
          <div style={{ marginTop: 16 }}>
            <Paragraph type="secondary" style={{ marginBottom: 8 }}>
              粘贴自定义 EXIF JSON（结构与 <code>template.json</code> 一致）
            </Paragraph>
            <Input.TextArea
              value={customJson}
              autoSize={{ minRows: 4, maxRows: 8 }}
              placeholder="在此粘贴或编辑自定义 EXIF JSON 配置"
              onChange={(v) => {
                setCustomJson(v);
                if (customJsonError) setCustomJsonError(null);
              }}
              status={customJsonError ? 'error' : undefined}
            />
            {customJsonError && (
              <div style={{ marginTop: 4, color: '#f53f3f', fontSize: 12 }}>
                {customJsonError}
              </div>
            )}
          </div>
        )}
      </Card>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 16,
          width: '100%',
          alignItems: 'flex-start',
        }}
      >
        <Card
          style={{
            flex: '1 1 420px',
            minWidth: 0,
            minHeight: 360,
            boxSizing: 'border-box',
          }}
          title="待处理图片"
          extra={
            <Space size={8}>
              <Upload
                showUploadList={false}
                multiple
                customRequest={handleUpload}
                accept="image/*"
              >
                <Button size="small" type="primary">上传图片</Button>
              </Upload>
              <Button type="text" size="small" onClick={handleClearUpload}>
                清空列表
              </Button>
            </Space>
          }
        >
          <Space direction="vertical" style={{ width: '100%' }} size={16}>
            {items.length === 0 && (
              <div
                style={{
                  height: 200,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#c9cdd4',
                }}
              >
                暂无上传图片，请先点击右上角按钮上传
              </div>
            )}
            {items.length > 0 && (
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 16,
                }}
              >
                {items.map((item) => {
                  const exifAll = (item.exif || {}) as Record<string, unknown>;
                  const zeroIfd = (exifAll['0th'] || {}) as Record<
                    string,
                    unknown
                  >;
                  const exifIfd = (exifAll.Exif || {}) as Record<string, unknown>;
                  const formatValue = (v: unknown): string => {
                    if (v == null) return '-';
                    if (Array.isArray(v)) {
                      return v.map((x) => String(x)).join(', ');
                    }
                    return String(v);
                  };
                  const make = formatValue(zeroIfd.Make);
                  const model = formatValue(zeroIfd.Model);
                  const lensModel = formatValue(exifIfd.LensModel);
                  const fNumber = formatValue(exifIfd.FNumber);
                  const exposureTime = formatValue(exifIfd.ExposureTime);
                  const isActive = activeItem && activeItem.id === item.id;
                  return (
                    <div
                      key={item.id}
                      onClick={() => setActiveId(item.id)}
                      onDoubleClick={() => {
                        setPreviewItem(item);
                        setPreviewVisible(true);
                      }}
                      style={{
                        width: 210,
                        borderRadius: 8,
                        border: isActive
                          ? '1px solid #165dff'
                          : '1px solid #e5e6eb',
                        backgroundColor: '#fff',
                        boxShadow: isActive
                          ? '0 0 0 2px rgba(22,93,255,0.08)'
                          : 'none',
                        cursor: 'pointer',
                        overflow: 'hidden',
                        position: 'relative',
                      }}
                    >
                      <div
                        style={{
                          height: 120,
                          overflow: 'hidden',
                          backgroundColor: '#f2f3f5',
                        }}
                      >
                        <img
                          src={`${baseUrl}/api/exif/view/upload/${item.id}`}
                          alt={item.filename}
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            display: 'block',
                          }}
                        />
                      </div>
                      <div
                        style={{
                          position: 'absolute',
                          top: 8,
                          right: 8,
                          padding: '2px 6px',
                          borderRadius: 999,
                          backgroundColor: item.aigc ? '#00b42a' : '#86909c',
                          color: '#fff',
                          fontSize: 10,
                        }}
                      >
                        {item.aigc ? 'AIGC' : '非 AIGC'}
                      </div>
                      <div style={{ padding: 8 }}>
                        <div
                          style={{
                            fontSize: 11,
                            color: '#4e5969',
                            marginBottom: 4,
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {item.filename}
                        </div>
                        <div
                          style={{
                            fontSize: 11,
                            color: '#86909c',
                            lineHeight: 1.4,
                          }}
                        >
                          <div>Make: {make}</div>
                          <div>Model: {model}</div>
                          <div>LensModel: {lensModel}</div>
                          <div>FNumber: {fNumber}</div>
                          <div>ExposureTime: {exposureTime}</div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Space>
        </Card>

        <Card
          style={{
            flex: '1 1 420px',
            minWidth: 0,
            minHeight: 360,
            boxSizing: 'border-box',
          }}
          title="处理结果"
          extra={
            <Space size={8}>
              <Button
                size="small"
                onClick={handleBatchDownload}
                disabled={!processedIds.length}
              >
                批量下载 ZIP
              </Button>
              <Button
                size="small"
                type="text"
                onClick={handleClearResults}
              >
                清空结果
              </Button>
              <Button
                size="small"
                type="text"
                onClick={handleDownload}
                disabled={!activeItem}
              >
                下载当前结果
              </Button>
            </Space>
          }
        >
          {processedIds.length ? (
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 16,
              }}
            >
              {items
                .filter((item) => processedIds.includes(item.id))
                .map((item) => (
                  <div
                    key={item.id}
                    style={{
                      width: 220,
                      borderRadius: 8,
                      border: '1px solid #e5e6eb',
                      backgroundColor: '#fff',
                      overflow: 'hidden',
                      cursor: 'pointer',
                    }}
                    onClick={() => {
                      setActiveId(item.id);
                      setResultPreviewItem(item);
                      setResultPreviewVisible(true);
                    }}
                  >
                    <div
                      style={{
                        height: 130,
                        overflow: 'hidden',
                        backgroundColor: '#f2f3f5',
                      }}
                    >
                      <img
                        src={`${baseUrl}/api/exif/view/output/${item.id}`}
                        alt={item.filename}
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover',
                          display: 'block',
                        }}
                      />
                    </div>
                    <div style={{ padding: 8 }}>
                      <div
                        style={{
                          fontSize: 11,
                          color: '#4e5969',
                          marginBottom: 4,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}
                      >
                        {item.filename}
                      </div>
                      <div
                        style={{
                          fontSize: 11,
                          color: '#86909c',
                          lineHeight: 1.4,
                        }}
                      >
                        <div>
                          尺寸:{' '}
                          {item.width && item.height
                            ? `${item.width} x ${item.height}`
                            : '-'}
                        </div>
                        <div>格式: {item.format || '-'}</div>
                        <div>AIGC: {item.aigc ? '是' : '否'}</div>
                        <div
                          style={{
                            marginTop: 8,
                            display: 'flex',
                            justifyContent: 'flex-end',
                          }}
                        >
                          <Button
                            size="mini"
                            type="text"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDownloadById(item.id);
                            }}
                          >
                            下载该结果
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          ) : (
            <div
              style={{
                height: 240,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#c9cdd4',
              }}
            >
              暂无处理结果
            </div>
          )}
        </Card>
      </div>

      <Modal
        visible={previewVisible}
        footer={null}
        onCancel={() => setPreviewVisible(false)}
        style={{ width: 960 }}
        title={
          previewItem
            ? `${previewItem.filename} · ${previewItem.format || ''} · ${
                previewItem.width || '-'
              } x ${previewItem.height || '-'}`
            : '图片预览'
        }
      >
        {previewItem && (
          <Space align="start" size={24} style={{ width: '100%' }}>
            <div
              style={{
                flex: 2,
                textAlign: 'center',
              }}
            >
              <img
                src={`${baseUrl}/api/exif/view/upload/${previewItem.id}`}
                alt={previewItem.filename}
                style={{
                  maxWidth: '100%',
                  maxHeight: '70vh',
                  borderRadius: 8,
                }}
              />
            </div>
            <div
              style={{
                flex: 1,
                maxHeight: '70vh',
                overflow: 'auto',
              }}
            >
              <Paragraph style={{ fontWeight: 500, marginBottom: 8 }}>
                基本信息
              </Paragraph>
              <Descriptions
                column={1}
                data={[
                  {
                    label: '文件名',
                    value: previewItem.filename,
                  },
                  {
                    label: '格式',
                    value: previewItem.format || '-',
                  },
                  {
                    label: '分辨率',
                    value:
                      previewItem.width && previewItem.height
                        ? `${previewItem.width} x ${previewItem.height}`
                        : '-',
                  },
                  {
                    label: 'AIGC',
                    value: previewItem.aigc ? '是' : '否',
                  },
                ]}
              />
              <Paragraph style={{ marginTop: 16, marginBottom: 8 }}>
                全部元数据
              </Paragraph>
              {renderAllMetadata(previewItem.exif)}
            </div>
          </Space>
        )}
      </Modal>

      <Modal
        visible={resultPreviewVisible}
        onCancel={() => setResultPreviewVisible(false)}
        style={{ width: 960 }}
        footer={
          resultPreviewItem ? (
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button
                onClick={() => handleDownloadById(resultPreviewItem.id)}
              >
                下载该结果
              </Button>
            </Space>
          ) : null
        }
        title={
          resultPreviewItem
            ? `${resultPreviewItem.filename} · ${resultPreviewItem.format || ''} · ${
                resultPreviewItem.width || '-'
              } x ${resultPreviewItem.height || '-'}`
            : '处理结果预览'
        }
      >
        {resultPreviewItem && (
          <Space align="start" size={24} style={{ width: '100%' }}>
            <div
              style={{
                flex: 2,
                textAlign: 'center',
              }}
            >
              <img
                src={`${baseUrl}/api/exif/view/output/${resultPreviewItem.id}`}
                alt={resultPreviewItem.filename}
                style={{
                  maxWidth: '100%',
                  maxHeight: '70vh',
                  borderRadius: 8,
                }}
              />
            </div>
            <div
              style={{
                flex: 1,
                maxHeight: '70vh',
                overflow: 'auto',
              }}
            >
              <Paragraph style={{ fontWeight: 500, marginBottom: 8 }}>
                基本信息
              </Paragraph>
              <Descriptions
                column={1}
                data={[
                  {
                    label: '文件名',
                    value: resultPreviewItem.filename,
                  },
                  {
                    label: '格式',
                    value: resultPreviewItem.format || '-',
                  },
                  {
                    label: '分辨率',
                    value:
                      resultPreviewItem.width && resultPreviewItem.height
                        ? `${resultPreviewItem.width} x ${resultPreviewItem.height}`
                        : '-',
                  },
                  {
                    label: 'AIGC',
                    value: resultPreviewItem.aigc ? '是' : '否',
                  },
                ]}
              />
              <Paragraph style={{ marginTop: 16, marginBottom: 4 }}>
                全部元数据
              </Paragraph>
              {renderAllMetadata(resultPreviewItem.exif)}
            </div>
          </Space>
        )}
      </Modal>
    </Space>
  );
};

export default ExifToolPage;
