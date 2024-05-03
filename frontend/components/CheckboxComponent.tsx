import { BsCheck } from 'react-icons/bs';
import styles from './form.module.css';

export interface CheckboxProps {
  label: string | JSX.Element;
  checked?: boolean;
  value?: string;
  onChange: () => void;
  id?: string;
  disabled?: boolean;
  isLoading?: boolean;
  icon?: JSX.Element;
}

const CheckboxComponent = ({
  label,
  checked,
  value,
  onChange,
  disabled,
  id,
  isLoading,
  icon,
}: CheckboxProps) => {
  const color = '#00B074';
  return (
    <div className={`flex cursor-pointer ${disabled ? 'opacity-[0.4]' : ''}`}>
      {icon ? (
        <div
          id={id || 'custom-checkbox-'}
          role={'checkbox'}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onChange();
          }}
          aria-checked={checked}
          aria-label="Check"
          className={`transition-all
          flex justify-center items-center w-[18px] h-[18px] rounded-[3px] bg-white  cursor-pointer`}
        >
          {icon}
        </div>
      ) : (
        <div
          id={id || 'custom-checkbox-'}
          role={'checkbox'}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onChange();
          }}
          aria-checked={checked}
          aria-label="Check"
          className={`transition-all ${styles['checkbox']} ${
            checked && styles['checkbox__checked']
          }`}
        >
          <BsCheck color={checked ? color : 'white'} />
        </div>
      )}

      <div className="ml-2 text-sm">
        <label
          htmlFor={value || 'custom-checkbox-'}
          className=" font-light font-beatrice cursor-pointer text-gray-900"
        >
          {label}
        </label>
      </div>
    </div>
  );
};

export default CheckboxComponent;
