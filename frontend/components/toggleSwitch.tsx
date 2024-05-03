import React from 'react';

export interface Props {
  checked?: any;
  checkHandler?: () => void;
  disabled?: boolean;
  active?: boolean;
}
const SwitchBtn = ({ checked, checkHandler, disabled }: Props) => {
  return (
    <div className="flex">
      <label>
        <span className="wrapper">
          <span className="checkboxHolder"></span>
          <input
            type="checkbox"
            checked={checked}
            onChange={checkHandler}
            disabled={disabled}
            style={{ display: 'none' }}
          />
        </span>
      </label>

      <style jsx>
        {`
          .wrapper {
            ${!checked
              ? `background-color: rgb(255, 255, 255); 
            border: 1px solid rgb(223, 223, 223); 
            border-radius: 15px; 
            box-shadow: rgb(223, 223, 223) 0px 0px 0px 0px inset; 
            box-sizing: border-box; 
            display: inline-block; 
            height: 18px; 
            opacity: 1; 
            position: relative; 
            user-select: none; 
            width: 30px; 
            transition: all 0.2s ease 0s;`
              : `background-color: #2ABB1E; 
            border: 1px solid #2ABB1E; 
            border-radius: 15px; 
            box-shadow: #2ABB1E 0px 0px 0px 20px inset; 
            box-sizing: border-box; 
            display: inline-block; 
            height: 18px; 
            opacity: 1; 
            position: relative; 
            user-select: none; 
            width: 32px;`}
          }

          .checkboxHolder {
            ${!checked
              ? `
              background-color: rgb(255, 255, 255); 
              border-radius: 100%; 
              box-shadow: rgba(0, 0, 0, 0.4) 0px 1px 3px; 
              cursor: pointer; 
              display: inline-block; 
              height: 15px; left: 0px; 
              position: absolute; top: 0px; 
              width: 15px; transition: all 0.2s ease 0s;`
              : `
              background-color: rgb(255, 255, 255); 
              border-radius: 100%; 
              box-shadow: rgba(0, 0, 0, 0.4) 0px 1px 3px; 
              cursor: pointer; 
              display: inline-block; 
              height: 15px; 
              left: 15px; 
              position: absolute; 
              top: 0px; 
              width: 15px;`}
          }
        `}
      </style>
    </div>
  );
};

export default SwitchBtn;
