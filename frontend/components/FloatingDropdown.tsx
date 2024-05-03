import KeydownArrow from '@/components/Svgs/chevron-down.svg';
import KeyupArrow from '@/components/Svgs/chevron-up.svg';
import React, { ReactNode, RefObject, useEffect, useRef } from 'react';

export interface FloatingDropdownProps {
  wrapperClassName?: string;
  icon?: React.ReactNode;
  title?: string | React.ReactNode;
children?: ReactNode | ((ref: RefObject<HTMLDivElement>) => ReactNode);
  activeClassName?: string;
  width?: string;
  className?: string;
  titleClassName?: string;
  titleWidth?: string;
  position?: string;
  disableDropdown?: boolean;
}

const FloatingDropdown = ({
  wrapperClassName,
  icon,
  title,
  children,
  activeClassName,
  titleClassName,
  className,
  position,
  width = 'min-w-[400px]',
  titleWidth,
  disableDropdown,
}: FloatingDropdownProps) => {
  const [isActive, setIsActive] = React.useState(false);

  const refOne = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.addEventListener(
      'click',
      (e: any) => {
        if (!refOne?.current?.contains?.(e.target)) {
          setIsActive(false);
        }
      },
      true
    );

    document.addEventListener('keyup', (e) => {
      if (e.code === 'Escape') {
        setIsActive(false);
      }
    });
  }, [refOne]);

  return (
    <div
      className={`${
        wrapperClassName ?? ``
      } transition-all duration-300 ${width} min-h-[20px] relative`}
      ref={refOne}
    >
      <div
        className={`${titleWidth ?? 'w-full'} bg-fail h-full ${
          isActive ? `rounded-t-2xl ${activeClassName}` : 'rounded-full'
        } px-2 cursor-pointer py-2 flex flex-row gap-3 items-center justify-between`}
        onClick={() => {
          if (disableDropdown) return;
          setIsActive(!isActive);
        }}
      >
        <div className="flex flex-row gap-3">
          <div
            className={`${
              titleClassName ?? 'text-white text-sm font-semibold'
            } `}
          >
            {title ?? 'Title'}
          </div>
        </div>

        <KeydownArrow
          className={`text-[lightgray] opacity-[0.2] text-bold ml-1 ${
            isActive ? 'hidden' : 'block'
          }`}
        />
        <KeyupArrow
          className={`text-[lightgray] opacity-[0.2] text-bold ml-1 ${
            !isActive ? 'hidden' : 'block'
          }`}
        />
      </div>

      <div
        className={` ${className} bg-white min-h-[50px] max-h-[300px] overflow-scroll scrollbar-hide ${
          position ?? 'absolute'
        } rounded-b-2xl shadow-2xl min-w-full ${
          isActive ? 'block border-[1px]' : 'hidden'
        }`}
        style={{
          zIndex: 1000,
        }}
      >
        {(typeof children === 'function' ? (children as any)?.(() => {
          setIsActive(false);
        }) : children)}
      </div>
    </div>
  );
};

export default FloatingDropdown;
