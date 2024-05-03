import React, { useState } from 'react';
import { BsChevronDown, BsChevronUp } from 'react-icons/bs';

export interface SlideDownAccordionProps {
  title?: React.ReactNode;
  children?: React.ReactNode;
  rightAddOn?: JSX.Element;
  disable?: boolean;
  containerClassName?: string;
}

/**
 *
 * @param title {String}
 * @param children {React.ReactNode}
 * @param rightAddOn: {JSX.Element}
 * @param disable: {Boolean}
 *
 * @returns
 */
const SlideDownAccordion = ({
  title,
  children,
  rightAddOn,
  disable,
  containerClassName,
}: SlideDownAccordionProps) => {
  const [active, setActive] = useState(disable ? true : false);
  return (
    <div
      className={`w-full font-beatrice  bg-white border-[1px] border-primary-10 shadow-sm rounded-lg flex flex-col transition-all gap-3 duration-500 p-4 justify-start ${
        active ? 'h-full' : ''
      }`}
    >
      <div
        className="flex flex-row justify-between"
        onClick={() => {
          if (disable) return;
          setActive(!active);
        }}
      >
        <div className="font-semibold text-lg">{title}</div>

        <div className="items-center flex flex-row gap-2">
          {rightAddOn}

          {disable ? (
            <></>
          ) : (
            <div className="">
              {active ? <BsChevronUp /> : <BsChevronDown />}
            </div>
          )}
        </div>
      </div>

      {!active ? (
        <></>
      ) : (
        <div
          className={`w-full flex flex-col items-start duration-100 transition-all flex-1 ${containerClassName}`}
        >
          {children}
        </div>
      )}
    </div>
  );
};

export default SlideDownAccordion;
