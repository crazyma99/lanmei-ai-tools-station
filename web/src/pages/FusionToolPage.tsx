import React, { useState } from 'react';
import {
  Card,
  Upload,
  Space,
  Button,
  Message,
  Image as ArcoImage,
  Typography,
  Switch,
  Spin,
  List,
} from '@arco-design/web-react';
import { IconDownload } from '@arco-design/web-react/icon';
import type { RequestOptions } from '@arco-design/web-react/es/Upload';
import axios from 'axios';
import './FusionToolPage.css';

const { Paragraph } = Typography;

interface FusionResponse {
  job_id: string;
  message: string;
  files: {
    png_url: string;
    jpg_url: string;
    watermark_url?: string | null;
  };
}

const FusionToolPage: React.FC = () => {
  const [fgFile, setFgFile] = useState<File | null>(null);
  const [bgFile, setBgFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FusionResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [enablePreview, setEnablePreview] = useState(false);
  const [watermark, setWatermark] = useState(false);
  const [fgPreviewUrl, setFgPreviewUrl] = useState<string | null>(null);
  const [bgPreviewUrl, setBgPreviewUrl] = useState<string | null>(null);

  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';
  const api = axios.create({
    baseURL: baseUrl,
  });

  const handleFgUpload = async (options: RequestOptions) => {
    const { file } = options;
    if (!file) return;
    const realFile = file as File;
    setFgFile(realFile);
    if (fgPreviewUrl) {
      URL.revokeObjectURL(fgPreviewUrl);
    }
    const localUrl = URL.createObjectURL(realFile);
    setFgPreviewUrl(localUrl);
    if (enablePreview) {
      await requestPreview(realFile);
    }
    if (options.onSuccess) {
      options.onSuccess();
    }
  };

  const handleBgUpload = async (options: RequestOptions) => {
    const { file } = options;
    if (!file) return;
    const realFile = file as File;
    setBgFile(realFile);
    if (bgPreviewUrl) {
      URL.revokeObjectURL(bgPreviewUrl);
    }
    const localUrl = URL.createObjectURL(realFile);
    setBgPreviewUrl(localUrl);
    if (options.onSuccess) {
      options.onSuccess();
    }
  };

  const requestPreview = async (file: File) => {
    setPreviewLoading(true);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await api.post('/api/fusion/preview', form, {
        responseType: 'blob',
      });
      const url = URL.createObjectURL(res.data);
      setPreviewUrl(url);
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } } };
      Message.error(err?.response?.data?.detail || '分割预览失败');
      setPreviewUrl(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleFusion = async () => {
    if (!fgFile || !bgFile) {
      Message.warning('请先上传前景与背景图片');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const form = new FormData();
      form.append('foreground', fgFile);
      form.append('background', bgFile);
      form.append('watermark', String(watermark));
      const res = await api.post<FusionResponse>('/api/fusion/process', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(res.data);
      Message.success('叠图完成');
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } } };
      Message.error(err?.response?.data?.detail || '叠图失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (type: 'png' | 'jpg' | 'wm') => {
    if (!result) return;
    const url =
      type === 'png'
        ? result.files.png_url
        : type === 'jpg'
        ? result.files.jpg_url
        : result.files.watermark_url;
    
    if (!url) {
      Message.warning('文件不存在');
      return;
    }
    const fullUrl = url.startsWith('http') ? url : `${baseUrl}${url}`;
    window.open(fullUrl, '_blank');
  };

  return (
    <div className="fusion-tool-page">
      <Space direction="vertical" style={{ width: '100%' }} size={16}>
        <div
        style={{
          display: 'flex',
          gap: 16,
          alignItems: 'stretch',
          width: '100%',
        }}
        >
          <div
          style={{
            flex: '0 0 260px',
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
          }}
          >
          <Card title="人物导入 / 导出">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              {fgPreviewUrl && (
                <ArcoImage
                  src={fgPreviewUrl}
                  alt="前景预览"
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: 260,
                    borderRadius: 8,
                    display: 'block',
                    margin: '0 auto',
                  }}
                />
              )}
              <Upload
                showUploadList={false}
                customRequest={handleFgUpload}
                accept="image/*"
              >
                <Button type="primary" long>
                  上传前景（人物）图
                </Button>
              </Upload>
            </Space>
          </Card>

          <Card title="底图 / 参考">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              {bgPreviewUrl && (
                <ArcoImage
                  src={bgPreviewUrl}
                  alt="背景预览"
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: 260,
                    borderRadius: 8,
                    display: 'block',
                    margin: '0 auto',
                  }}
                />
              )}
              <Upload
                showUploadList={false}
                customRequest={handleBgUpload}
                accept="image/*"
              >
                <Button long>上传背景图</Button>
              </Upload>
            </Space>
          </Card>

          <Card title="BiSeNet 分割预览">
            <Space
              direction="vertical"
              size={8}
              style={{ width: '100%', alignItems: 'center' }}
            >
              <Space>
                <span>启用分割预览</span>
                <Switch
                  checked={enablePreview}
                  onChange={async (v: boolean) => {
                    setEnablePreview(v);
                    if (v && fgFile) {
                      await requestPreview(fgFile);
                    } else {
                      setPreviewUrl(null);
                    }
                  }}
                />
              </Space>
              {previewLoading ? (
                 <div style={{ padding: 20, textAlign: 'center' }}>
                   <Spin tip="正在生成预览..." />
                 </div>
              ) : previewUrl ? (
                <ArcoImage
                  src={previewUrl}
                  alt="分割预览"
                  style={{
                    width: '100%',
                    height: 'auto',
                    maxHeight: 260,
                    borderRadius: 8,
                    display: 'block',
                    margin: '0 auto',
                  }}
                />
              ) : (
                <Paragraph type="secondary">暂无分割预览</Paragraph>
              )}
            </Space>
          </Card>
          
          <div style={{ padding: '0 8px' }}>
             <Space>
                <span>添加水印</span>
                <Switch checked={watermark} onChange={setWatermark} />
             </Space>
          </div>

          <Button
            type="primary"
            loading={loading}
            onClick={handleFusion}
            style={{ marginTop: 4 }}
          >
            开始叠图
          </Button>
        </div>

        <div
          style={{
            flex: '1 1 auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
          }}
        >
          <Card title="状态信息">
            {loading ? (
               <div style={{ textAlign: 'center', padding: 20 }}>
                 <Spin tip="正在进行图像融合处理，请稍候..." />
               </div>
            ) : result ? (
              <Paragraph>{result.message}</Paragraph>
            ) : (
              <Paragraph type="secondary">
                暂无处理结果，请先上传图片并点击开始叠图
              </Paragraph>
            )}
          </Card>

          <Card title="对比预览 A（前 / 后）">
            <div
              style={{
                display: 'flex',
                gap: 16,
                width: '100%',
                minHeight: 420,
              }}
            >
              <div
                style={{
                  flex: 1,
                  minWidth: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <Paragraph type="secondary">原始前景 / 分割参考</Paragraph>
                <div
                  style={{
                    flex: 1,
                    borderRadius: 8,
                    overflow: 'hidden',
                    backgroundColor: '#000',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {fgPreviewUrl || previewUrl ? (
                    <ArcoImage
                      src={previewUrl || fgPreviewUrl || ''}
                      alt="前景 / 分割参考"
                      style={{
                        maxWidth: '100%',
                        maxHeight: '100%',
                        width: 'auto',
                        height: 'auto',
                        objectFit: 'contain',
                        display: 'block',
                      }}
                    />
                  ) : (
                    <Paragraph type="secondary">请先上传前景图并开启分割预览</Paragraph>
                  )}
                </div>
              </div>
              <div
                style={{
                  flex: 1,
                  minWidth: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <Paragraph type="secondary">叠图结果</Paragraph>
                <div
                  style={{
                    flex: 1,
                    borderRadius: 8,
                    overflow: 'hidden',
                    backgroundColor: '#000',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {result ? (
                    <ArcoImage
                      src={(() => {
                        const url = result.files.watermark_url || result.files.jpg_url;
                        return url.startsWith('http') ? url : `${baseUrl}${url}`;
                      })()}
                      alt="叠图结果"
                      style={{
                        maxWidth: '100%',
                        maxHeight: '100%',
                        width: 'auto',
                        height: 'auto',
                        objectFit: 'contain',
                        display: 'block',
                      }}
                    />
                  ) : (
                    <Paragraph type="secondary">暂无结果</Paragraph>
                  )}
                </div>
              </div>
            </div>
          </Card>

          <Card title="下载结果 (PNG / JPG)">
            {result ? (
              <List
                size="small"
                header="处理完成的文件"
                dataSource={[
                  { type: 'png', url: result.files.png_url, label: 'PNG 格式' },
                  { type: 'jpg', url: result.files.jpg_url, label: 'JPG 格式' },
                  ...(result.files.watermark_url
                    ? [
                        {
                          type: 'wm',
                          url: result.files.watermark_url,
                          label: '水印版 (JPG)',
                        },
                      ]
                    : []),
                ]}
                render={(item: any) => (
                  <List.Item
                    key={item.type}
                    actions={[
                      <Button
                        type="text"
                        size="small"
                        icon={<IconDownload />}
                        onClick={() => handleDownload(item.type)}
                      >
                        下载
                      </Button>,
                    ]}
                  >
                    <List.Item.Meta
                      title={item.url.split('/').pop()}
                      description={item.label}
                    />
                  </List.Item>
                )}
              />
            ) : (
              <Paragraph type="secondary">暂无可下载文件</Paragraph>
            )}
          </Card>
        </div>
      </div>
      </Space>
    </div>
  );
};

export default FusionToolPage;
