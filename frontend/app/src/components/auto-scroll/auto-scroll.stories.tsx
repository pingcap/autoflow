import type { Meta, StoryObj } from '@storybook/react';
import { type RefObject, useEffect, useRef, useState } from 'react';
import { AutoScroll } from './auto-scroll';
import { ManualScrollVoter } from './manual-scroll-voter';
import { useRequestScroll } from './use-request-scroll';

const meta = {
  title: 'Components/AutoScroll',
  subcomponents: {},
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {},
  args: {},
} satisfies Meta<any>;

export default meta;

type Story = StoryObj<typeof meta>

export const Container: Story = {
  args: {},
  render: () => {
    const [count, setCount] = useState(0);
    const [target, setTarget] = useState<HTMLDivElement | null>(null);

    useEffect(() => {
      const interval = setInterval(() => {
        setCount(count => count + 1);
      }, 1000);

      return () => {
        clearInterval(interval);
      };
    }, []);

    return (
      <div ref={setTarget} style={{ minWidth: 250, height: 500, overflow: 'scroll' }}>
        <AutoScroll target={target}>
          <ManualScrollVoter />
          <AutoScrollByObservingChildren target={{ current: target }} />
          {Array(count).fill(0).map((_, i) => (
            <div key={i} style={{ margin: 20, padding: 20, height: 200, background: '#c00000' }}></div>
          ))}
        </AutoScroll>
      </div>
    );
  },
};

export const Document: Story = {
  args: {},
  render: () => {
    const [count, setCount] = useState(0);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
      const interval = setInterval(() => {
        setCount(count => count + 1);
      }, 1000);

      return () => {
        clearInterval(interval);
      };
    }, []);

    return (
      <AutoScroll>
        <ManualScrollVoter />
        <AutoScrollByObservingChildren target={ref} />
        <div ref={ref}>
          {Array(count).fill(0).map((_, i) => (
            <div key={i} style={{ margin: 20, padding: 20, height: 200, background: '#c00000' }}></div>
          ))}
        </div>
      </AutoScroll>
    );
  },
};

function AutoScrollByObservingChildren ({ target }: { target: RefObject<HTMLElement | null> }) {
  const requestScroll = useRequestScroll();

  useEffect(() => {
    const el = target.current;
    if (el) {

      const mo = new MutationObserver(() => {
        requestScroll('bottom');
      });

      mo.observe(el, { childList: true });

      return () => {
        mo.disconnect();
      };
    }
  }, [target.current]);

  return null;
}
