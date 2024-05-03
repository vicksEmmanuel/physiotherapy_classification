import React from 'react';

const Circle = ({ isActive }: { isActive?: boolean }) => {
  return (
    <div className="rounded-full w-[15px] h-[15px] relative border-primary items-center justify-center flex border-[1px] bg-white">
      <div
        className={`rounded-full w-[70%] h-[70%] ${
          isActive ? 'bg-primary' : 'bg-white'
        }`}
      />
    </div>
  );
};

export default Circle;
