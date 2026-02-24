import React, { useState } from 'react';
import { Layout, Menu } from '@arco-design/web-react';
import { IconImage, IconCamera } from '@arco-design/web-react/icon';
import ExifToolPage from './pages/ExifToolPage.tsx';
import FusionToolPage from './pages/FusionToolPage.tsx';
import '@arco-design/web-react/dist/css/arco.css';
import './App.css';

const { Sider, Header, Content } = Layout;

const App: React.FC = () => {
  const [activeKey, setActiveKey] = useState<'exif' | 'fusion'>('exif');

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider breakpoint="lg" collapsible>
        <div style={{ padding: 16, color: '#fff', fontWeight: 600 }}>
          Lanmei AI Tools
        </div>
        <Menu
          selectedKeys={[activeKey]}
          onClickMenuItem={(key: string) => setActiveKey(key as 'exif' | 'fusion')}
          style={{ height: '100%' }}
        >
          <Menu.Item key="exif">
            <IconImage /> EXIF 工具
          </Menu.Item>
          <Menu.Item key="fusion">
            <IconCamera /> 人像智能叠图
          </Menu.Item>
        </Menu>
      </Sider>
      <Layout>
        <Header style={{ padding: '0 24px', fontSize: 16 }}>
          {activeKey === 'exif' ? 'EXIF 清洗 / 修改 / AIGC 检测' : 'AI 智能人像叠图'}
        </Header>
        <Content style={{ padding: 24, overflow: 'auto' }}>
          {activeKey === 'exif' ? <ExifToolPage /> : <FusionToolPage />}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
